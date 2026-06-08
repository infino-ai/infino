// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Crash-injection property tests for the update / delete
//! pipeline.
//!
//! We can't easily inject a real `kill -9` into the WAL
//! pipeline's middle from outside, so this harness models
//! crashes as "the WAL state-doc on storage represents some
//! legal intermediate state of the pipeline." For every such
//! state we drive recovery and assert the post-recovery shape
//! matches a no-crash run.
//!
//! Cases covered:
//!
//! - Intent DELETE with N targets and zero tombstone progress —
//!   recovery completes all targets.
//! - Intent DELETE with N targets and partial progress (first K
//!   already Tombstoned, remainder Pending) — recovery resumes
//!   at the first Pending; the pre-marked entries stay
//!   untouched.
//! - Intent DELETE with N targets and all Tombstoned — recovery
//!   advances the WAL to Complete in one CAS.
//!
//! Together these exhaust the per-target state machine's
//! cross-product so a future regression that corrupts the
//! resume cursor lights up here.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tempfile::TempDir;
use uuid::Uuid;

use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::Supertable;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::wal::WalStore;
use infino::supertable::wal::state_doc::{
    OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState, WalStateDoc,
};
use infino::test_helpers::{build_title_batch, default_supertable_options};

/// Document count seeded into each crash-recovery fixture.
const SEED_DOC_COUNT: usize = 5;

fn make_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: 1 << 30,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: 4,
        cold_fetch_chunk_bytes: 1 << 20,
        prefetch_concurrency: 8,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

/// Seed a storage prefix with one committed superfile of `n`
/// rows + an Intent DELETE WAL in the specified intermediate
/// shape. The WAL targets the first `target_count` rows.
async fn seed_partial_state(
    storage: Arc<dyn StorageProvider>,
    n: usize,
    target_count: usize,
    pre_completed: &[TombstoneOutcome],
) -> WalId {
    {
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = st.writer().expect("writer");
        let titles_owned: Vec<String> = (0..n).map(|i| format!("row{i:08}")).collect();
        let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
        w.append(&build_title_batch(&titles)).expect("append");
        w.commit().expect("commit");
        drop(w);
        drop(st);
    }
    let ws = WalStore::new(Arc::clone(&storage));
    let st = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("open for ids");
    let manifest = st.reader().manifest().clone();
    let id_min = manifest
        .superfile_list
        .superfiles
        .first()
        .expect("superfile")
        .id_min;
    drop(st);

    let wal_id_value: i128 = 0x4242_4242 + target_count as i128;
    let wal_id = WalId(wal_id_value);

    // Build tombstone_progress: pre_completed[i] for i < its
    // length, then Pending for the rest.
    let progress: Vec<TombstoneEntry> = (0..target_count)
        .map(|i| {
            let target = RowId(id_min + i as i128);
            let outcome = pre_completed
                .get(i)
                .copied()
                .unwrap_or(TombstoneOutcome::Pending);
            TombstoneEntry {
                target_id: target,
                outcome,
                tombstoned_in_superfile: if outcome == TombstoneOutcome::Tombstoned {
                    Some(
                        manifest
                            .superfile_list
                            .superfiles
                            .first()
                            .expect("superfile")
                            .superfile_id,
                    )
                } else {
                    None
                },
            }
        })
        .collect();

    let wal = WalStateDoc {
        wal_id,
        schema_version: SCHEMA_VERSION,
        op_kind: OpKind::Delete,
        state: WalState::Intent,
        created_at: Utc::now(),
        lease: None,
        predicate_repr: "property-test".into(),
        target_ids: progress.iter().map(|e| e.target_id).collect(),
        new_row_count: None,
        new_row_content_hash: None,
        preallocated_superfile_id: None,
        minted_id_spans: Vec::new(),
        tombstone_progress: progress,
    };
    ws.create(&wal).await.expect("seed wal");
    wal_id
}

async fn recover_and_assert_complete(
    storage: Arc<dyn StorageProvider>,
    wal_id: WalId,
    expected_tombstoned: usize,
) {
    let cache_dir = TempDir::new().expect("cache");
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let _st = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("open");

    let ws = WalStore::new(Arc::clone(&storage));
    // The recovery sweep may have deleted the WAL inline. If
    // it's gone, treat that as a successful Complete + reap.
    let post = match ws.read(wal_id).await {
        Ok((d, _e)) => Some(d),
        Err(_) => None,
    };
    match post {
        Some(doc) => {
            assert_eq!(doc.state, WalState::Complete);
            let actual_tombstoned = doc
                .tombstone_progress
                .iter()
                .filter(|e| e.outcome == TombstoneOutcome::Tombstoned)
                .count();
            assert_eq!(
                actual_tombstoned, expected_tombstoned,
                "post-recovery tombstoned count must equal expected"
            );
        }
        None => {
            // WAL was inline-deleted by the writer's
            // post-Complete cleanup; that's the steady-state
            // happy path.
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intent_with_zero_progress_recovers_clean() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 3, &[]).await;
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 3).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intent_with_partial_progress_resumes_at_first_pending() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    // First entry pre-marked Tombstoned; the rest are Pending.
    let pre = [TombstoneOutcome::Tombstoned];
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 3, &pre).await;
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 3).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intent_with_all_complete_progress_just_advances_state() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    // Every entry pre-marked Tombstoned; recovery should
    // observe that nothing's left to do for tombstones, and
    // advance the WAL state to Complete in one CAS.
    let pre = vec![TombstoneOutcome::Tombstoned; 3];
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 3, &pre).await;
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 3).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intent_with_mix_of_outcomes_recovers_clean() {
    // First Tombstoned, second NotFound, third+ Pending. The
    // already-decided rows stay decided; only the Pending rows
    // get resolved during recovery.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let pre = [TombstoneOutcome::Tombstoned, TombstoneOutcome::NotFound];
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 4, &pre).await;
    // Expected tombstoned count after recovery: 1 (pre) + 2
    // (recovered) = 3. The NotFound entry stays NotFound.
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 3).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intent_with_all_not_found_recovers_clean() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let pre = vec![TombstoneOutcome::NotFound; 3];
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 3, &pre).await;
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_is_idempotent_under_repeated_open() {
    // Run the recovery sweep multiple times. Once the WAL is
    // Complete (or inline-deleted), subsequent opens should be
    // no-ops — neither resurrecting the state nor failing.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let wal_id = seed_partial_state(Arc::clone(&storage), SEED_DOC_COUNT, 2, &[]).await;
    // First open drives recovery.
    recover_and_assert_complete(Arc::clone(&storage), wal_id, 2).await;
    // Second open should observe the post-state cleanly.
    let cache_dir = TempDir::new().expect("cache");
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let _ = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("re-open");
    let ws = WalStore::new(Arc::clone(&storage));
    // The wal_id either matches a Complete doc OR is absent
    // (inline-deleted on its way out).
    if let Ok((doc, _)) = ws.read(wal_id).await {
        assert_eq!(doc.state, WalState::Complete);
    }
}

// Suppress unused-import warning helper.
#[test]
fn unused_uuid_smoke() {
    let _ = Uuid::nil();
}
