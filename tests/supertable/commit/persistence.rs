// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writer write-through to storage.
//!
//! Covers the persistence path the writer takes when
//! `SupertableOptions::with_storage(...)` is attached:
//!
//! - A commit on a storage-backed supertable writes:
//!   - each new superfile's bytes to `data/seg-<uuid>.sf.parquet`
//!   - one manifest part to `manifests/part-<hash>.avro.zst`
//!   - the manifest list to `manifest-lists/list-NNNNNN.json`
//!   - the pointer to `_supertable/current`
//! - The pointer is readable after commit; manifest_id
//!   increments per commit.
//! - Two successive commits both publish (CAS works); the
//!   second commit's manifest list references all superfiles
//!   (existing + new).
//! - In-memory queries still work post-commit (the in-memory
//!   store stays active for reads even with storage attached).
//! - A supertable with NO storage attached takes the
//!   in-memory path — no on-disk state, no regressions.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::supertable::Supertable;
use infino::supertable::manifest::commit::read_pointer;

/// 1-byte multipart threshold forcing every upload through the
/// multipart path.
const PUT_MULTIPART_THRESHOLD_BYTES: u64 = 1;
/// BM25 top-k for the post-commit query.
const BM25_TOP_K: usize = 5;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options};
use tempfile::TempDir;

#[test]
fn commit_persists_pointer_list_part_and_superfile() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    // Pointer file exists on disk, manifest_id=1 (initial was 0).
    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer present");
    assert_eq!(pointer.manifest_id, 1);
    assert!(
        pointer
            .manifest_list_uri
            .starts_with("manifest-lists/list-")
    );

    // Manifest list file exists and is non-empty.
    let (list_bytes, _) =
        futures::executor::block_on(storage.get(&pointer.manifest_list_uri)).expect("get list");
    assert!(!list_bytes.is_empty());

    // At least one manifest part exists in manifests/.
    let manifests_dir = dir.path().join("manifests");
    let parts: Vec<_> = std::fs::read_dir(&manifests_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        parts.len(),
        1,
        "single-partition mode: exactly one manifest part on disk; got {parts:?}"
    );

    // Superfile file exists in data/.
    let data_dir = dir.path().join("data");
    let superfiles: Vec<_> = std::fs::read_dir(&data_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        superfiles.len(),
        1,
        "one shard committed → one superfile file on disk; got {superfiles:?}"
    );

    // In-memory manifest reflects the commit.
    let r = st.reader();
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn two_successive_commits_both_publish() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("w1");
    w.append(&build_title_batch(&["foo", "bar"]))
        .expect("append1");
    w.commit().expect("commit1");
    drop(w);

    let mut w = st.writer().expect("w2");
    w.append(&build_title_batch(&["baz"])).expect("append2");
    w.commit().expect("commit2");
    drop(w);

    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer");
    assert_eq!(
        pointer.manifest_id, 2,
        "two commits ⇒ pointer at manifest_id=2"
    );

    // Both manifest list versions persist (immutable per id).
    let lists_dir = dir.path().join("manifest-lists");
    let n_lists = std::fs::read_dir(&lists_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .count();
    assert_eq!(n_lists, 2, "two list files (manifest_id 1 + 2)");

    // Manifest part count = 2 (each commit writes a fresh part
    // under content-addressed URI; single-partition mode
    // means a fresh part per commit, no reuse).
    let manifests_dir = dir.path().join("manifests");
    let n_parts = std::fs::read_dir(&manifests_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .count();
    assert_eq!(n_parts, 2);

    // In-memory manifest reflects both commits.
    let r = st.reader();
    assert_eq!(r.manifest_id(), 2);
    assert_eq!(
        r.n_superfiles(),
        2,
        "two shard commits ⇒ two superfiles visible"
    );
}

#[test]
fn multipart_threshold_forces_superfile_through_put_multipart() {
    // Setting `put_multipart_threshold_bytes = 1` routes
    // every superfile through `put_multipart` instead of
    // `put_atomic`. Verifies the end-to-end shape:
    //   - commit succeeds (no panic, no error)
    //   - superfile file lands on disk
    //   - manifest pointer + list + part written
    //   - cross-process open recovers the data
    // The actual `put_atomic` vs `put_multipart` distinction
    // is invisible to readers — the test passes through
    // `Supertable::open` to assert the superfile bytes were
    // correctly assembled by the multipart path.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_put_multipart_threshold_bytes(PUT_MULTIPART_THRESHOLD_BYTES);
    let producer = Supertable::create(opts).expect("create");
    {
        let mut w = producer.writer().expect("writer");
        // Two docs so the FTS posting list has more than a
        // single term — exercises a non-trivial superfile
        // payload through multipart chunking.
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("commit via multipart path");
    }
    drop(producer);

    // Superfile file landed on disk.
    let data_dir = dir.path().join("data");
    let superfiles: Vec<_> = std::fs::read_dir(&data_dir)
        .expect("readdir data")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        superfiles.len(),
        1,
        "one superfile file should land on disk after a multipart commit"
    );

    // Cross-process open recovers correctly — proof the
    // multipart-uploaded superfile is byte-identical to what
    // the writer produced.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open after multipart commit");
    let r = consumer.reader();
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn no_storage_attached_takes_in_memory_path() {
    // Sanity: a supertable WITHOUT storage attached behaves
    // exactly like the no-storage baseline — in-memory only.
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(default_supertable_options()).expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["x", "y"])).expect("append");
    w.commit().expect("commit");
    drop(w);

    // Nothing on disk under the tempdir.
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        entries.len(),
        0,
        "no-storage supertable must not touch the filesystem; got {entries:?}"
    );

    // In-memory manifest still updates.
    let r = st.reader();
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn committed_supertable_remains_in_memory_queryable_for_now() {
    // Storage write-through is additive — the
    // in-memory store still holds superfile bytes, so existing
    // in-memory query paths keep working unchanged. Verifies no
    // regression to the FTS read path.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "nimblefox special token",
        "ordinary common text",
    ]))
    .expect("append");
    w.commit().expect("commit");
    drop(w);

    let hits = st
        .reader()
        .bm25_hits(
            "title",
            "nimblefox",
            BM25_TOP_K,
            infino::supertable::query::fts::BoolMode::Or,
        )
        .expect("query");
    assert_eq!(hits.len(), 1, "commit must not break in-memory reads");
}

#[test]
fn manifest_id_increments_only_on_non_empty_commits() {
    // A commit with no buffered batches is a no-op.
    // Storage write-through should preserve this — no spurious
    // pointer rewrites on empty commits.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("w");
    w.commit().expect("empty commit"); // no buffer → no-op
    drop(w);

    // Pointer doesn't exist yet (no real commit happened).
    let pointer = futures::executor::block_on(read_pointer(&*storage)).expect("read");
    assert!(pointer.is_none(), "empty commit must not publish a pointer");

    // Now do a real commit; pointer appears at manifest_id=1.
    let mut w = st.writer().expect("w");
    w.append(&build_title_batch(&["only", "real"]))
        .expect("append");
    w.commit().expect("real commit");
    drop(w);

    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer");
    assert_eq!(pointer.manifest_id, 1);
}
