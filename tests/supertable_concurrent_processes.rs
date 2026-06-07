// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OCC on the manifest pointer (cross-process commits).
//!
//! Two children fork, each commits concurrently, and the test
//! verifies the manifest-id sequence + final state.
//! This file simulates the cross-process scenario via two
//! independent `Supertable` handles sharing the same on-disk
//! storage. Each handle owns its own `writer_outstanding`
//! slot + in-memory state, so they're observationally
//! equivalent to two processes: the only synchronization
//! point is the conditional-write commit on the shared
//! pointer file.
//!
//! Coverage:
//!  - Two handles racing on the pointer → OCC retry loop in
//!    `writer::persist_commit` retries the loser, both end
//!    up committed at monotonic manifest_ids (1, 2).
//!  - Three handles racing → retry handles cascading
//!    contention; final manifest_id == 3.
//!  - Loser sees the winner's superfiles after retry
//!    (the refresh-inside-retry path's
//!    inherit-via-content-addressed-Arc::clone is exercised).

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::supertable::Supertable;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options};

/// Sentinel retry count asserted to plumb through the options builder.
const MAX_RETRIES_SENTINEL: u32 = 42;
/// Generous retry budget so contending commits eventually all land.
const CONTENTION_MAX_RETRIES: u32 = 20;
use tempfile::TempDir;

/// Two independent handles racing to commit. The OCC retry
/// loop must ensure both commits eventually succeed and the
/// final pointer sits at manifest_id = 2 with both writers'
/// superfiles visible.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_handles_concurrent_commits_both_succeed_via_occ_retry() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    // Commit via spawn_blocking — SupertableWriter::commit is
    // sync, and its persist_commit internally detects the
    // ambient runtime via Handle::current().
    let t_a = tokio::task::spawn_blocking({
        let st = st_a.clone();
        move || {
            let mut w = st.writer().expect("writer A");
            w.append(&build_title_batch(&["from_a alpha"]))
                .expect("append A");
            w.commit().expect("commit A");
        }
    });
    let t_b = tokio::task::spawn_blocking({
        let st = st_b.clone();
        move || {
            let mut w = st.writer().expect("writer B");
            w.append(&build_title_batch(&["from_b beta"]))
                .expect("append B");
            w.commit().expect("commit B");
        }
    });

    t_a.await.expect("task A");
    t_b.await.expect("task B");

    // The handle that won the first pointer-CAS is at
    // manifest_id = 1; the loser retried after refreshing and
    // is at manifest_id = 2. We can't assert which was which
    // (the race is non-deterministic), but the max must be 2.
    let final_ids = [st_a.manifest_id(), st_b.manifest_id()];
    let max_id = final_ids.iter().copied().max().expect("non-empty");
    let min_id = final_ids.iter().copied().min().expect("non-empty");
    assert_eq!(
        max_id, 2,
        "one handle must have committed at v2 after retry; got {final_ids:?}"
    );
    assert_eq!(
        min_id, 1,
        "both commits succeeded so both handles advanced past v0; got {final_ids:?}"
    );

    // Fresh open against the same storage sees the union.
    drop(st_a);
    drop(st_b);
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    assert_eq!(consumer.manifest_id(), 2);
    assert_eq!(
        consumer.reader().n_superfiles(),
        2,
        "post-open consumer sees both writers' superfiles"
    );
}

/// Three handles racing — exercises cascading retries (the
/// second loser may itself lose to the first loser's retry
/// before winning at manifest_id = 3).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_handles_concurrent_commits_all_succeed() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_c = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let t_a = tokio::task::spawn_blocking({
        let st = st_a.clone();
        move || {
            let mut w = st.writer().expect("writer A");
            w.append(&build_title_batch(&["from_a"])).expect("append A");
            w.commit().expect("commit A");
        }
    });
    let t_b = tokio::task::spawn_blocking({
        let st = st_b.clone();
        move || {
            let mut w = st.writer().expect("writer B");
            w.append(&build_title_batch(&["from_b"])).expect("append B");
            w.commit().expect("commit B");
        }
    });
    let t_c = tokio::task::spawn_blocking({
        let st = st_c.clone();
        move || {
            let mut w = st.writer().expect("writer C");
            w.append(&build_title_batch(&["from_c"])).expect("append C");
            w.commit().expect("commit C");
        }
    });

    t_a.await.expect("task A");
    t_b.await.expect("task B");
    t_c.await.expect("task C");

    drop(st_a);
    drop(st_b);
    drop(st_c);
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    assert_eq!(
        consumer.manifest_id(),
        3,
        "three concurrent commits must result in manifest_id = 3"
    );
    assert_eq!(consumer.reader().n_superfiles(), 3);

    // Verify all three writers' superfiles are present by
    // counting distinct segment URIs — the supertable injects
    // `_id` values via its monotonic generator, so we can't
    // assert specific id values across writer processes (each
    // gets its own random worker_id).
    let reader = consumer.reader();
    let segs = &reader.manifest().superfile_list.superfiles;
    let uris: std::collections::HashSet<_> = segs.iter().map(|s| s.uri.0).collect();
    assert_eq!(
        uris.len(),
        segs.len(),
        "every segment carries a distinct URI"
    );
    assert!(
        segs.len() >= 3,
        "expected ≥ 3 superfiles from three writers; got {}",
        segs.len()
    );
}

/// After a retry-driven commit, the loser's in-memory state
/// must reflect both its own superfiles AND the winner's. This
/// is the inherit-via-content-addressed-Arc::clone path —
/// `refresh_inner_state_async` running inside the retry loop
/// ArcSwaps the winner's manifest into `inner.manifest`
/// before the next attempt's `with_appended` chains the
/// loser's superfiles on top.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retry_winner_sees_loser_segments_in_final_manifest() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let t_a = tokio::task::spawn_blocking({
        let st = st_a.clone();
        move || {
            let mut w = st.writer().expect("writer A");
            w.append(&build_title_batch(&["alpha"])).expect("append A");
            w.commit().expect("commit A");
        }
    });
    let t_b = tokio::task::spawn_blocking({
        let st = st_b.clone();
        move || {
            let mut w = st.writer().expect("writer B");
            w.append(&build_title_batch(&["beta"])).expect("append B");
            w.commit().expect("commit B");
        }
    });

    t_a.await.expect("task A");
    t_b.await.expect("task B");

    // Whichever handle ended at manifest_id = 2 should also
    // see 2 superfiles — its own plus the winner's. The handle
    // at manifest_id = 1 only sees its own segment (pre-retry
    // state, never refreshed).
    for st in [&st_a, &st_b] {
        let r = st.reader();
        if r.manifest_id() == 2 {
            assert_eq!(
                r.n_superfiles(),
                2,
                "v2 handle must see both superfiles (its own + winner's)"
            );
        } else if r.manifest_id() == 1 {
            assert_eq!(
                r.n_superfiles(),
                1,
                "v1 handle (winner) sees only its own segment"
            );
        } else {
            panic!("unexpected manifest_id: {}", r.manifest_id());
        }
    }
}

/// Sequential commits across two handles — the second
/// handle's first commit reads the persisted pointer + parts
/// and chains its commit at manifest_id = 2 without ever
/// hitting contention. Sanity check that the contended-commit
/// path degenerates cleanly to the non-contended case when no
/// race occurs.
#[test]
fn sequential_commits_across_handles_no_retry_needed() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Handle A commits first.
    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    {
        let mut w = st_a.writer().expect("writer A");
        w.append(&build_title_batch(&["first"])).expect("append A");
        w.commit().expect("commit A");
    }
    assert_eq!(st_a.manifest_id(), 1);
    drop(st_a);

    // Handle B opens (sees A's state), then commits.
    let st_b = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("open B");
    assert_eq!(st_b.manifest_id(), 1, "B opens at A's last manifest_id");
    {
        let mut w = st_b.writer().expect("writer B");
        w.append(&build_title_batch(&["second"])).expect("append B");
        w.commit().expect("commit B");
    }
    assert_eq!(st_b.manifest_id(), 2);
    assert_eq!(
        st_b.reader().n_superfiles(),
        2,
        "B sees both A's and B's superfiles"
    );
}

/// `with_max_commit_retries` plumbs through to the writer's
/// OCC retry loop. With retries=1, the first lost CAS
/// surfaces `WriteContentionExhausted` immediately; with
/// retries=20 (above the default), the same two-handle race
/// still resolves (writer just has more headroom).
///
/// The "retries=1 always fails on contention" property is
/// timing-sensitive (the second commit might not race; the
/// first one will succeed without conflict). So this test
/// asserts only the structural property: the field round-
/// trips through `SupertableOptions::with_max_commit_retries`
/// + a high value lets concurrent commits succeed the same
/// way the default does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_commit_retries_is_plumbed_through_options() {
    // Builder roundtrip.
    {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let opts = default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_max_commit_retries(MAX_RETRIES_SENTINEL);
        assert_eq!(opts.max_commit_retries, MAX_RETRIES_SENTINEL);
    }

    // Concurrent commit with raised retries — same outcome
    // as the default-retries test (both commits succeed,
    // final manifest_id = 2), proves the knob doesn't
    // break the path.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st_a = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_max_commit_retries(CONTENTION_MAX_RETRIES),
    )
    .expect("create");
    let st_b = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_max_commit_retries(CONTENTION_MAX_RETRIES),
    )
    .expect("create");

    let t_a = tokio::task::spawn_blocking({
        let st = st_a.clone();
        move || {
            let mut w = st.writer().expect("writer A");
            w.append(&build_title_batch(&["alpha"])).expect("append A");
            w.commit().expect("commit A");
        }
    });
    let t_b = tokio::task::spawn_blocking({
        let st = st_b.clone();
        move || {
            let mut w = st.writer().expect("writer B");
            w.append(&build_title_batch(&["beta"])).expect("append B");
            w.commit().expect("commit B");
        }
    });

    t_a.await.expect("task A");
    t_b.await.expect("task B");

    drop(st_a);
    drop(st_b);
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    assert_eq!(consumer.manifest_id(), 2);
}
