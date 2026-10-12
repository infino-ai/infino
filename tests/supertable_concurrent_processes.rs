// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OCC on the manifest pointer (cross-process commits).
//!
//! Independent `Supertable` handles over one shared storage stand in
//! for separate processes: each owns its writer slot and in-memory
//! manifest, so the conditional pointer write is the only point where
//! their commits meet.
//!
//! Manifest ids can skip under contention: a retry never reuses an id
//! whose manifest list already exists, even its own unpublished one.
//! Tests assert ordering and superfile counts, never exact ids.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use infino::{
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;
use tokio::task::spawn_blocking;

/// Sentinel retry count asserted to plumb through the options builder.
const MAX_RETRIES_SENTINEL: u32 = 42;
/// Generous retry budget so contending commits eventually all land.
const CONTENTION_MAX_RETRIES: u32 = 20;

/// Commits one row from each handle concurrently and waits for all of them.
///
/// Commits run on blocking threads because `SupertableWriter::commit` is
/// sync and drives the ambient runtime itself.
async fn commit_concurrently(handles: &[&Supertable]) {
    let tasks: Vec<_> = handles
        .iter()
        .enumerate()
        .map(|(writer, st)| {
            let st = (*st).clone();
            spawn_blocking(move || {
                let title = format!("from writer {writer}");
                let mut w = st.writer().expect("writer");
                w.append(&build_title_batch(&[title.as_str()]))
                    .expect("append");
                w.commit().expect("commit");
            })
        })
        .collect();
    for task in tasks {
        task.await.expect("commit task");
    }
}

/// Opens a fresh handle on `storage`, seeing every published commit.
fn reopen(storage: &Arc<dyn StorageProvider>) -> Supertable {
    Supertable::open(default_supertable_options().with_storage(Arc::clone(storage))).expect("open")
}

/// Two handles racing: both commits land, the loser past the winner.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_handles_concurrent_commits_both_succeed_via_occ_retry() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    commit_concurrently(&[&st_a, &st_b]).await;

    // Which handle wins is non-deterministic.
    let final_ids = [st_a.manifest_id(), st_b.manifest_id()];
    let max_id = final_ids.iter().copied().max().expect("non-empty");
    let min_id = final_ids.iter().copied().min().expect("non-empty");
    assert!(
        min_id >= 1,
        "both handles must have advanced past v0; got {final_ids:?}"
    );
    assert!(
        max_id > min_id,
        "the loser must have committed past the winner; got {final_ids:?}"
    );

    let consumer = reopen(&storage);
    assert_eq!(consumer.manifest_id(), max_id);
    assert_eq!(
        consumer.reader().expect("reader").n_superfiles(),
        2,
        "a fresh open sees both writers' superfiles"
    );
}

/// Three handles racing: a loser can lose again to another loser's
/// retry, and every commit still lands.
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

    commit_concurrently(&[&st_a, &st_b, &st_c]).await;

    let consumer = reopen(&storage);
    assert!(
        consumer.manifest_id() >= 3,
        "three commits must advance the pointer at least three times; got {}",
        consumer.manifest_id()
    );
    let reader = consumer.reader().expect("reader");
    let superfiles = reader.manifest().get_all_superfiles();
    assert_eq!(superfiles.len(), 3, "one superfile per writer");
    let uris: HashSet<_> = superfiles.iter().map(|s| s.uri.0).collect();
    assert_eq!(uris.len(), 3, "every writer's superfile is distinct");
}

/// The loser's retry builds on the winner's manifest, so its in-memory
/// state holds both superfiles; the winner never refreshed and holds only
/// its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retry_loser_holds_winner_superfiles() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    commit_concurrently(&[&st_a, &st_b]).await;

    // The higher manifest_id committed last. `pinned_reader` reads each
    // handle's in-memory manifest; `reader` would refresh the winner from
    // storage and hide the difference.
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
        "the loser holds both superfiles"
    );
    assert_eq!(
        winner.pinned_reader().n_superfiles(),
        1,
        "the winner holds only its own superfile"
    );
}

/// Sequential commits across two handles never contend: the second
/// handle opens at the first one's manifest and commits straight after it.
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
    let st_b = reopen(&storage);
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

/// `with_max_commit_retries` reaches the options, and a raised budget
/// still resolves a two-handle race.
///
/// A budget of 1 is not tested failing: whether the commits actually
/// collide is timing-dependent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_commit_retries_is_plumbed_through_options() {
    let opts = default_supertable_options().with_max_commit_retries(MAX_RETRIES_SENTINEL);
    assert_eq!(opts.max_commit_retries, MAX_RETRIES_SENTINEL);

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

    commit_concurrently(&[&st_a, &st_b]).await;

    let consumer = reopen(&storage);
    assert!(
        consumer.manifest_id() >= 2,
        "both commits must advance the pointer; got {}",
        consumer.manifest_id()
    );
    assert_eq!(consumer.reader().expect("reader").n_superfiles(), 2);
}
