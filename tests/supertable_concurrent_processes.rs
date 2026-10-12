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
//!    up committed at distinct, increasing manifest_ids.
//!  - Three handles racing → retry handles cascading
//!    contention; all three commits land.
//!  - Loser sees the winner's superfiles after retry
//!    (the refresh-inside-retry path's
//!    inherit-via-content-addressed-Arc::clone is exercised).
//!
//! Manifest ids are not dense under contention: a retry skips any
//! id whose list object exists above the pointer (a crash orphan or
//! a racing writer's unpublished list, possibly its own), so the
//! final id may exceed the commit count. Tests assert ordering and
//! superfile counts, never exact ids.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::{
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};

/// Sentinel retry count asserted to plumb through the options builder.
const MAX_RETRIES_SENTINEL: u32 = 42;
/// Generous retry budget so contending commits eventually all land.
const CONTENTION_MAX_RETRIES: u32 = 20;
use tempfile::TempDir;

/// Two independent handles racing to commit. The OCC retry
/// loop must ensure both commits eventually succeed and the
/// final pointer shows both writers' superfiles.
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

    // The loser retried after refreshing, so it sits past the
    // winner; which handle won is non-deterministic.
    let final_ids = [st_a.manifest_id(), st_b.manifest_id()];
    let max_id = final_ids.iter().copied().max().expect("non-empty");
    let min_id = final_ids.iter().copied().min().expect("non-empty");
    assert!(
        min_id >= 1,
        "both commits succeeded so both handles advanced past v0; got {final_ids:?}"
    );
    assert!(
        max_id > min_id,
        "the loser must have committed past the winner; got {final_ids:?}"
    );

    // Fresh open against the same storage sees the union.
    drop(st_a);
    drop(st_b);
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    assert_eq!(consumer.manifest_id(), max_id);
    assert_eq!(
        consumer.reader().expect("reader").n_superfiles(),
        2,
        "post-open consumer sees both writers' superfiles"
    );
}

/// Three handles racing — exercises cascading retries (the
/// second loser may itself lose to the first loser's retry
/// before winning).
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
    assert!(
        consumer.manifest_id() >= 3,
        "three concurrent commits must advance the pointer at least three times; got {}",
        consumer.manifest_id()
    );
    assert_eq!(consumer.reader().expect("reader").n_superfiles(), 3);

    // Verify all three writers' superfiles are present by
    // counting distinct superfile URIs — the supertable injects
    // `_id` values via its monotonic generator, so we can't
    // assert specific id values across writer processes (each
    // gets its own random worker_id).
    let reader = consumer.reader().expect("reader");
    let segs = reader.manifest().get_all_superfiles();
    let uris: std::collections::HashSet<_> = segs.iter().map(|s| s.uri.0).collect();
    assert_eq!(
        uris.len(),
        segs.len(),
        "every superfile carries a distinct URI"
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
async fn retry_winner_sees_loser_superfiles_in_final_manifest() {
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

    // The handle at the higher manifest_id committed last, so its
    // in-memory manifest holds both superfiles — its own plus the
    // winner's. The winner never refreshed, so it holds only its own.
    // `pinned_reader` skips the freshness check `reader` applies,
    // which would pull the loser's commit into the winner too.
    let (loser, winner) = if st_a.manifest_id() > st_b.manifest_id() {
        (&st_a, &st_b)
    } else {
        (&st_b, &st_a)
    };
    assert!(
        loser.manifest_id() > winner.manifest_id(),
        "the loser must have committed past the winner"
    );
    assert_eq!(
        loser.pinned_reader().n_superfiles(),
        2,
        "the loser must see both superfiles (its own + winner's)"
    );
    assert_eq!(
        winner.pinned_reader().n_superfiles(),
        1,
        "the winner sees only its own superfile"
    );
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
        st_b.reader().expect("reader").n_superfiles(),
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
    // as the default-retries test (both commits succeed),
    // proves the knob doesn't break the path.
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
    assert!(
        consumer.manifest_id() >= 2,
        "both commits must advance the pointer; got {}",
        consumer.manifest_id()
    );
    assert_eq!(consumer.reader().expect("reader").n_superfiles(), 2);
}
