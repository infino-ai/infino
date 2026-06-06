// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Recovery-sweep white-box tests.
//!
//! These drive `Supertable::open` against a pre-seeded
//! `wal/mutations/` prefix and assert on WAL state documents and
//! lease ownership directly — internal state that is not part of the
//! public API — so they live in-crate rather than under `tests/`.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use tempfile::TempDir;

use crate::storage::{LocalFsStorageProvider, StorageProvider};
use crate::supertable::Supertable;
use crate::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use crate::supertable::wal::WalStore;
use crate::supertable::wal::state_doc::{
    OpKind, RowId, SCHEMA_VERSION, SupertableHandleId, TombstoneEntry, TombstoneOutcome, WalId,
    WalState, WalStateDoc,
};
use crate::test_helpers::{build_title_batch, default_supertable_options};

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

fn seed_intent_delete_wal(target_id: i128, wal_id_v: i128) -> WalStateDoc {
    WalStateDoc {
        wal_id: WalId(wal_id_v),
        schema_version: SCHEMA_VERSION,
        op_kind: OpKind::Delete,
        state: WalState::Intent,
        created_at: Utc::now(),
        lease: None,
        predicate_repr: "recovery test".into(),
        target_ids: vec![RowId(target_id)],
        new_row_count: None,
        new_row_content_hash: None,
        preallocated_superfile_id: None,
        minted_id_spans: Vec::new(),
        tombstone_progress: vec![TombstoneEntry {
            target_id: RowId(target_id),
            outcome: TombstoneOutcome::Pending,
            tombstoned_in_superfile: None,
        }],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_time_sweep_drives_pre_seeded_intent_walls_to_complete() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // 1. Phase one: create a supertable, commit a superfile, drop
    //    it. The pointer + superfile bytes are durable now.
    {
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha", "beta", "gamma"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        drop(st);
    }

    // 2. Phase two: pre-seed an Intent DELETE WAL targeting the
    //    first row's `_id`. The sweep should pick it up and
    //    advance it to Complete.
    let ws = WalStore::new(Arc::clone(&storage));
    let target_id;
    {
        let st = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
        let manifest = st.reader().manifest().clone();
        target_id = manifest
            .superfile_list
            .superfiles
            .first()
            .expect("superfile present")
            .id_min;
    }
    let wal = seed_intent_delete_wal(target_id, 0x1234_5678);
    ws.create(&wal).await.expect("seed wal");

    // 3. Re-open the supertable with a disk cache attached so
    //    follow-up reader queries can fault the superfile bytes
    //    in. The open-time sweep should drive the seeded WAL to
    //    Complete.
    let cache_dir = TempDir::new().expect("cache_dir");
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("re-open");
    let (post, _etag) = ws.read(wal.wal_id).await.expect("read after sweep");
    assert_eq!(post.state, WalState::Complete);
    assert_eq!(
        post.tombstone_progress[0].outcome,
        TombstoneOutcome::Tombstoned
    );
    // The tombstone bit is in the sidecar; a follow-up FTS query
    // against the same handle excludes the row.
    let hits = st
        .bm25_search(
            "title",
            "alpha",
            10,
            crate::superfile::fts::reader::BoolMode::Or,
        )
        .expect("fts");
    // The "alpha" row is local doc_id 0 — verify it's filtered.
    for hit in &hits {
        assert_ne!(hit.local_doc_id, 0);
    }
}

#[test]
fn create_with_existing_pointer_delegates_to_open() {
    // The point of `Supertable::create`'s create-or-open
    // shape: when storage already carries a committed pointer,
    // `create` MUST behave like `open` rather than silently
    // shadowing existing data with an empty manifest.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Phase 1: write some rows + drop the supertable so the
    // pointer file is durable on storage.
    {
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["one", "two", "three"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        drop(st);
    }

    // Phase 2: call `create` again against the same storage.
    // The pointer file is present, so `create` should
    // delegate to `open`'s load path and surface the committed
    // manifest — three rows visible.
    let cache_dir = TempDir::new().expect("cache");
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create with existing pointer");
    let manifest = st.reader().manifest().clone();
    assert!(
        !manifest.superfile_list.superfiles.is_empty(),
        "create against existing pointer must load the committed manifest"
    );
    let batches = st
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    let total = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count column")
        .value(0);
    assert_eq!(total, 3, "create-or-open must surface 3 committed rows");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_preempts_expired_lease_and_completes_wal() {
    // Simulates "process A died mid-pipeline holding the lease".
    // We seed an Intent DELETE WAL with a `lease` that's already
    // expired in wall-clock terms; a fresh `Supertable::open` in
    // this process sees the expired lease, preempts via the
    // recovery sweep, drives the WAL to Complete, and the
    // tombstone-phase outcome matches a no-crash run.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Phase 1: commit a superfile so the DELETE WAL has a
    // target to resolve.
    {
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["foo", "bar"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        drop(st);
    }

    // Phase 2: stamp the WAL state doc with an expired lease.
    let ws = WalStore::new(Arc::clone(&storage));
    let target_id;
    {
        let st = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open for manifest");
        let manifest = st.reader().manifest().clone();
        target_id = manifest
            .superfile_list
            .superfiles
            .first()
            .expect("superfile")
            .id_min;
        drop(st);
    }
    let now = Utc::now();
    let mut wal = seed_intent_delete_wal(target_id, 0xCAFE_BABE);
    wal.lease = Some(crate::supertable::wal::state_doc::Lease {
        // "Process A": some random owner id that's no longer
        // alive.
        owner: SupertableHandleId(0xDEAD_BEEF),
        acquired_at: now - chrono::Duration::seconds(600),
        expires_at: now - chrono::Duration::seconds(60),
    });
    ws.create(&wal).await.expect("seed");

    // Phase 3: open this process's supertable. The sweep
    // preempts the expired lease and drives the WAL to
    // Complete.
    let cache_dir = TempDir::new().expect("cache");
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("open after expired lease");

    // The WAL is now Complete; the new lease owner is the
    // current handle's id.
    let (post, _etag) = ws.read(wal.wal_id).await.expect("read");
    assert_eq!(post.state, WalState::Complete);
    assert_eq!(
        post.tombstone_progress[0].outcome,
        TombstoneOutcome::Tombstoned
    );
    let post_lease = post.lease.expect("lease set");
    assert_eq!(
        post_lease.owner,
        st.handle_id(),
        "this handle should own the lease after preemption"
    );
    // FTS query no longer returns the tombstoned row.
    let hits = st
        .bm25_search(
            "title",
            "foo",
            10,
            crate::superfile::fts::reader::BoolMode::Or,
        )
        .expect("fts");
    for hit in &hits {
        assert_ne!(hit.local_doc_id, 0);
    }
}
