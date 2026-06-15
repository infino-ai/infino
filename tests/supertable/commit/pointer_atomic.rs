// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Atomic-rename pointer commit.
//!
//! Covers the persistence primitives shipped by
//! `manifest::commit`:
//!
//! - Pointer file round-trip (text-format serde).
//! - Initial commit on a fresh supertable (no prev pointer)
//!   writes part(s) + list + pointer + nothing else.
//! - Second commit with a valid prev pointer succeeds.
//! - Second commit with a STALE prev etag surfaces
//!   `CommitError::WriteContentionExhausted` (the OCC
//!   contention signal the writer retries on).
//! - **Part reuse**: a commit with `parts_to_write: []`
//!   writes the manifest list + pointer but NO part files
//!   (zero `put_atomic` calls into the parts namespace).
//! - **Idempotent content-addressed part PUT**: writing a
//!   part whose content already exists at the same URI
//!   swallows `PreconditionFailed` cleanly.
//! - **Parallel-issue verification**: a barrier(2) mock
//!   storage proves the list PUT and the part PUT are issued
//!   in parallel — a serial implementation would deadlock at
//!   the barrier.

#![deny(clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::{Barrier, Mutex};
use uuid::Uuid;

use infino::supertable::manifest::commit::{
    self, MANIFEST_LISTS_DIR, MANIFEST_PARTS_DIR, POINTER_PATH, PointerFile, list_uri, part_uri,
    read_pointer, write_pointer,
};
use infino::supertable::manifest::list::{
    FORMAT_VERSION as LIST_FORMAT_VERSION, ManifestList, ManifestListEntry, PartitionStrategy,
};
use infino::supertable::manifest::part::{self as part_mod, ContentHash, ManifestPart, PartId};
use infino::supertable::storage::{
    LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider,
};
use infino::supertable::{CommitError, ManifestLoadError};
use tempfile::TempDir;

/// Manifest id used by the pointer-file round-trip fixture.
const POINTER_ROUNDTRIP_MANIFEST_ID: u64 = 42;
/// Manifest id used by the forward-compat parse fixture.
const POINTER_FORWARD_COMPAT_MANIFEST_ID: u64 = 7;
/// Byte filling a 32-byte fixture content hash.
const FIXTURE_CONTENT_HASH_BYTE: u8 = 0xab;
/// Default hash-partition bucket count in the manifest-list fixture.
const DEFAULT_HASH_N_BUCKETS: u32 = 64;
/// Number of leading PUTs (list + part) that gate on the barrier
/// before the pointer PUT in the parallel-commit test.
const PARALLEL_PUT_COUNT_BEFORE_POINTER: usize = 2;
/// Expected total PUT count for one commit: list + part + pointer.
const EXPECTED_COMMIT_PUT_COUNT: usize = 3;
/// Watchdog deadline (seconds) for the parallel-PUT poll loop.
const PARALLEL_PUT_POLL_TIMEOUT_SECS: u64 = 5;
/// Poll interval (ms) for the parallel-PUT loop.
const PARALLEL_PUT_POLL_INTERVAL_MS: u64 = 10;

async fn commit_manifest(
    storage: &dyn StorageProvider,
    expected_prev_etag: Option<&str>,
    new_list: &ManifestList,
    parts: &[&ManifestPart],
) -> Result<PointerFile, CommitError> {
    let encoded: Vec<Vec<u8>> = parts.iter().map(|p| part_mod::encode(p, 3)).collect();
    let encoded_refs: Vec<&[u8]> = encoded.iter().map(|b| b.as_slice()).collect();
    commit::commit_manifest(storage, expected_prev_etag, new_list, &encoded_refs).await
}

// ============================================================
// Pointer-file format
// ============================================================

#[test]
fn pointer_file_text_format_roundtrip() {
    let p = PointerFile {
        manifest_id: POINTER_ROUNDTRIP_MANIFEST_ID,
        manifest_list_uri: "manifest-lists/list-000042.json".into(),
        content_hash: ContentHash([FIXTURE_CONTENT_HASH_BYTE; 32]),
    };
    let bytes = p.to_bytes();
    let s = std::str::from_utf8(&bytes).expect("utf-8");
    assert!(
        s.contains("manifest_id=42"),
        "must spell out manifest_id; got {s:?}"
    );
    assert!(s.contains("manifest_list_uri=manifest-lists/list-000042.json"));
    assert!(s.contains("content_hash=blake3:"));
    let parsed = PointerFile::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed, p);
}

#[test]
fn pointer_file_rejects_truncated() {
    let bad = b"manifest_id=1\nmanifest_list_uri=foo\n"; // missing content_hash
    let err = PointerFile::from_bytes(bad).expect_err("must reject");
    assert!(matches!(err, ManifestLoadError::PointerParse(_)), "{err:?}");
}

#[test]
fn pointer_file_tolerates_unknown_keys_for_forward_compat() {
    let s = b"manifest_id=7\n\
              manifest_list_uri=manifest-lists/list-000007.json\n\
              content_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n\
              future_field=whatever\n";
    let p = PointerFile::from_bytes(s).expect("parse");
    assert_eq!(p.manifest_id, POINTER_FORWARD_COMPAT_MANIFEST_ID);
}

// ============================================================
// End-to-end commit against LocalFs
// ============================================================

fn fresh_part(seed: u8) -> ManifestPart {
    ManifestPart {
        format_version: part_mod::FORMAT_VERSION.into(),
        part_id: PartId(Uuid::from_bytes([seed; 16])),
        superfiles: vec![],
    }
}

fn empty_list(manifest_id: u64, parts: Vec<ManifestListEntry>) -> ManifestList {
    ManifestList {
        format_version: LIST_FORMAT_VERSION.into(),
        manifest_id,
        options_hash: ContentHash([0u8; 32]),
        schema: Vec::new(),
        id_column: "doc_id".into(),
        fts_columns: vec![],
        vector_columns: vec![],
        partition_strategy: PartitionStrategy::Hash {
            column: "doc_id".into(),
            n_buckets: DEFAULT_HASH_N_BUCKETS,
        },
        parts,
    }
}

/// Build a manifest list entry referencing an already-encoded
/// part. Skip-summary aggregates left empty here.
fn entry_for(part: &ManifestPart) -> ManifestListEntry {
    let encoded = part_mod::encode(part, 3);
    let hash = ContentHash::of(&encoded);
    let uri = part_uri(&hash);
    let size_compressed = encoded.len() as u64;
    let size_uncompressed = zstd::stream::decode_all(encoded.as_slice())
        .expect("self-decode")
        .len() as u64;
    ManifestListEntry {
        part_id: part.part_id,
        uri,
        n_superfiles: part.superfiles.len() as u64,
        size_bytes_compressed: size_compressed,
        size_bytes_uncompressed: size_uncompressed,
        content_hash: hash,
        partition_key: Vec::new(),
        id_range: (0, 0),
        scalar_stats_agg: Default::default(),
        fts_summary_agg: Default::default(),
        vector_summary_agg: Default::default(),
    }
}

#[tokio::test]
async fn initial_commit_writes_list_part_pointer() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");

    let part = fresh_part(1);
    let list = empty_list(0, vec![entry_for(&part)]);

    let pointer = commit_manifest(&storage, None, &list, &[&part])
        .await
        .expect("initial commit");

    assert_eq!(pointer.manifest_id, 0);
    assert_eq!(pointer.manifest_list_uri, list_uri(0));

    // Pointer is readable.
    let (read, _) = read_pointer(&storage).await.expect("read").expect("some");
    assert_eq!(read, pointer);
    // List + part are at their expected URIs.
    let (list_bytes, _) = storage.get(&list_uri(0)).await.expect("list bytes");
    assert!(!list_bytes.is_empty());
    let (part_bytes, _) = storage
        .get(&entry_for(&part).uri)
        .await
        .expect("part bytes");
    assert!(!part_bytes.is_empty());
}

#[tokio::test]
async fn no_prior_pointer_is_none() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
    let read = read_pointer(&storage).await.expect("read");
    assert!(read.is_none(), "fresh supertable has no pointer yet");
}

#[tokio::test]
async fn second_commit_with_valid_prev_etag_succeeds() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");

    let part_v0 = fresh_part(2);
    let list_v0 = empty_list(0, vec![entry_for(&part_v0)]);
    commit_manifest(&storage, None, &list_v0, &[&part_v0])
        .await
        .expect("v0");
    let etag_v0 = storage
        .head(POINTER_PATH)
        .await
        .expect("head v0")
        .etag
        .expect("etag");

    let part_v1 = fresh_part(3);
    let list_v1 = empty_list(1, vec![entry_for(&part_v0), entry_for(&part_v1)]);

    // Part-reuse: parts_to_write contains only the NEW part.
    // The previously-written part is just referenced by URI.
    let pointer = commit_manifest(&storage, Some(&etag_v0), &list_v1, &[&part_v1])
        .await
        .expect("v1");
    assert_eq!(pointer.manifest_id, 1);

    let (read, _) = read_pointer(&storage).await.expect("read").expect("some");
    assert_eq!(read.manifest_id, 1);
}

#[tokio::test]
async fn stale_prev_etag_surfaces_write_contention_exhausted() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");

    let part_v0 = fresh_part(4);
    let list_v0 = empty_list(0, vec![entry_for(&part_v0)]);
    commit_manifest(&storage, None, &list_v0, &[&part_v0])
        .await
        .expect("v0");
    let etag_v0 = storage
        .head(POINTER_PATH)
        .await
        .expect("head")
        .etag
        .expect("etag");

    // Legitimate v1 publishes.
    let part_v1 = fresh_part(5);
    let list_v1 = empty_list(1, vec![entry_for(&part_v0), entry_for(&part_v1)]);
    commit_manifest(&storage, Some(&etag_v0), &list_v1, &[&part_v1])
        .await
        .expect("v1");

    // Stale writer tries to publish with v0's etag — must fail.
    let part_v1_stale = fresh_part(6);
    let list_v1_stale = empty_list(1, vec![entry_for(&part_v1_stale)]);
    let err = commit_manifest(&storage, Some(&etag_v0), &list_v1_stale, &[&part_v1_stale])
        .await
        .expect_err("stale etag must fail");
    assert!(
        matches!(err, CommitError::WriteContentionExhausted),
        "expected WriteContentionExhausted, got {err:?}"
    );
}

#[tokio::test]
async fn part_reuse_writes_zero_new_part_files() {
    // Setup: v0 with one part already published.
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
    let part = fresh_part(7);
    let list_v0 = empty_list(0, vec![entry_for(&part)]);
    commit_manifest(&storage, None, &list_v0, &[&part])
        .await
        .expect("v0");

    // Snapshot the parts directory before v1.
    let parts_dir = dir.path().join(MANIFEST_PARTS_DIR);
    let count_before = std::fs::read_dir(&parts_dir).expect("readdir").count();

    // v1 reuses the same part — parts_to_write is empty. List
    // is rewritten (manifest_id changed) but no new part file
    // hits disk.
    let etag_v0 = storage
        .head(POINTER_PATH)
        .await
        .expect("head")
        .etag
        .expect("etag");
    let list_v1 = empty_list(1, vec![entry_for(&part)]);
    commit_manifest(&storage, Some(&etag_v0), &list_v1, &[])
        .await
        .expect("v1 no new parts");

    let count_after = std::fs::read_dir(&parts_dir).expect("readdir").count();
    assert_eq!(
        count_after, count_before,
        "part-reuse commit must write zero new part files \
         (before={count_before}, after={count_after})"
    );

    // But manifest_id 1 is published.
    let (read, _) = read_pointer(&storage).await.expect("read").expect("some");
    assert_eq!(read.manifest_id, 1);
}

#[tokio::test]
async fn idempotent_content_addressed_part_put() {
    // Direct test of write_manifest_part: writing the same
    // logical part twice (same bytes, same content hash, same
    // URI) succeeds both times. The second call swallows
    // PreconditionFailed because the content is identical.
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
    let part = fresh_part(9);

    let r1 = commit::write_manifest_part(&storage, &part, 3)
        .await
        .expect("first write");
    let r2 = commit::write_manifest_part(&storage, &part, 3)
        .await
        .expect("second write (idempotent)");
    assert_eq!(r1.uri, r2.uri);
    assert_eq!(r1.content_hash, r2.content_hash);
}

// ============================================================
// Parallel-issue verification via Barrier(2) mock
// ============================================================

/// Mock storage that funnels every write through a shared
/// `Barrier(2)`. A serial implementation issues two PUTs
/// sequentially → only one caller hits the barrier at a time
/// → deadlock. A parallel implementation issues both PUTs at
/// once → barrier opens → both PUTs complete.
///
/// Deterministic, not wall-clock-based — runs identically on
/// any CI environment.
#[derive(Debug)]
struct BarrierMockStorage {
    barrier: Arc<Barrier>,
    objects: Mutex<HashMap<String, Bytes>>,
    put_calls: AtomicUsize,
}

impl BarrierMockStorage {
    fn new(barrier_n: usize) -> Arc<Self> {
        Arc::new(Self {
            barrier: Arc::new(Barrier::new(barrier_n)),
            objects: Mutex::new(HashMap::new()),
            put_calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl StorageProvider for BarrierMockStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        let objs = self.objects.lock().await;
        match objs.get(uri) {
            Some(b) => Ok(ObjectMeta {
                size: b.len() as u64,
                etag: Some("mock-etag".into()),
            }),
            None => Err(StorageError::NotFound { uri: uri.into() }),
        }
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let objs = self.objects.lock().await;
        match objs.get(uri) {
            Some(b) => Ok((
                b.clone(),
                ObjectMeta {
                    size: b.len() as u64,
                    etag: Some("mock-etag".into()),
                },
            )),
            None => Err(StorageError::NotFound { uri: uri.into() }),
        }
    }

    async fn get_range(
        &self,
        _uri: &str,
        _range: std::ops::Range<u64>,
    ) -> Result<Bytes, StorageError> {
        Err(StorageError::Permanent {
            uri: "barrier-mock".into(),
            source: "get_range unused".into(),
        })
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        // Tokio's Barrier is reusable, but only opens when
        // exactly N parties have called wait() in the same
        // cycle. For this test we want the first 2 PUTs
        // (list + part) to gate each other at the barrier
        // (proving they're issued in parallel); subsequent
        // PUTs (the pointer) must bypass — otherwise a lone
        // third caller would deadlock waiting for a second
        // party that never arrives.
        let prior = self.put_calls.fetch_add(1, Ordering::AcqRel);
        if prior < PARALLEL_PUT_COUNT_BEFORE_POINTER {
            self.barrier.wait().await;
        }
        let mut objs = self.objects.lock().await;
        if objs.contains_key(uri) {
            return Err(StorageError::PreconditionFailed { uri: uri.into() });
        }
        objs.insert(uri.into(), bytes);
        Ok(Some("mock-etag".into()))
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        _expected: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let prior = self.put_calls.fetch_add(1, Ordering::AcqRel);
        if prior < PARALLEL_PUT_COUNT_BEFORE_POINTER {
            self.barrier.wait().await;
        }
        let mut objs = self.objects.lock().await;
        objs.insert(uri.into(), bytes);
        Ok(Some("mock-etag".into()))
    }

    async fn put_multipart(
        &self,
        _uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        Err(StorageError::Permanent {
            uri: "barrier-mock".into(),
            source: "put_multipart unused".into(),
        })
    }

    async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commit_issues_list_and_part_in_parallel() {
    // Two writes (list + one part) before the pointer commit.
    // Pointer commit comes AFTER the parallel issue; it's the
    // visibility barrier, not part of the parallel set. So we
    // need to: (a) barrier(2) for the parallel phase, then
    // (b) let the pointer through. The mock's barrier handles
    // the parallel phase; for the third put (the pointer) we
    // need the barrier to be reusable OR to detect "third
    // call, just let it through."
    //
    // Simpler shape: wrap the barrier check in "first 2 calls
    // wait on barrier(2); subsequent calls go through
    // immediately." Implement via the existing AtomicUsize.
    //
    // Below we sidestep by constructing the test with
    // barrier(3) and having the test itself await the barrier
    // alongside the commit — 2 storage PUTs + 1 test thread =
    // 3 parties. The test then verifies the storage saw both
    // parallel PUTs.
    //
    // …but that's fragile. Use the simpler invariant: spawn
    // commit_manifest on a tokio task with a barrier(2) on
    // put_atomic, then assert that within a short timeout
    // both put_atomic calls have arrived. If commit_manifest
    // were serial, only one would arrive.

    let storage = BarrierMockStorage::new(PARALLEL_PUT_COUNT_BEFORE_POINTER);
    let storage_dyn: Arc<dyn StorageProvider> = storage.clone();
    let part = fresh_part(20);
    let list = empty_list(0, vec![entry_for(&part)]);

    // Spawn the commit on a separate task so the test can
    // observe the put_calls counter while it's running.
    let commit_handle = {
        let storage_dyn = Arc::clone(&storage_dyn);
        let part = part.clone();
        let list = list.clone();
        tokio::spawn(
            async move { commit_manifest(storage_dyn.as_ref(), None, &list, &[&part]).await },
        )
    };

    // Wait for both PUTs (list + part) to arrive at the
    // barrier. If commit_manifest serialized them, only one
    // would arrive and we'd deadlock — the test would time
    // out via the timeout below.
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(PARALLEL_PUT_POLL_TIMEOUT_SECS);
    loop {
        if storage.put_calls.load(Ordering::Acquire) >= PARALLEL_PUT_COUNT_BEFORE_POINTER {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "parallel-issue verification failed: only {} put calls \
                 arrived at the barrier within 5s — commit_manifest \
                 appears to be serial",
                storage.put_calls.load(Ordering::Acquire)
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(
            PARALLEL_PUT_POLL_INTERVAL_MS,
        ))
        .await;
    }

    // Both PUTs arrived in parallel → barrier opens →
    // commit_manifest completes successfully (modulo the
    // pointer PUT which is a third put_calls hit; allowed
    // because barrier(2) is reusable on the next .wait()).
    let pointer = commit_handle.await.expect("join").expect("commit");
    assert_eq!(pointer.manifest_id, 0);
    // Total: 2 PUTs (list+part) + 1 PUT (pointer) = 3.
    assert_eq!(
        storage.put_calls.load(Ordering::Acquire),
        EXPECTED_COMMIT_PUT_COUNT
    );
}

// ============================================================
// write_pointer direct tests
// ============================================================

#[tokio::test]
async fn write_pointer_initial_then_update() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");

    let p0 = PointerFile {
        manifest_id: 0,
        manifest_list_uri: list_uri(0),
        content_hash: ContentHash([FIXTURE_CONTENT_HASH_BYTE; 32]),
    };
    write_pointer(&storage, &p0, None).await.expect("initial");

    let etag = storage
        .head(POINTER_PATH)
        .await
        .expect("head")
        .etag
        .expect("etag");

    let p1 = PointerFile {
        manifest_id: 1,
        manifest_list_uri: list_uri(1),
        content_hash: ContentHash([0xcd; 32]),
    };
    write_pointer(&storage, &p1, Some(&etag))
        .await
        .expect("update");

    let (read, _) = read_pointer(&storage).await.expect("read").expect("some");
    assert_eq!(read, p1);
}

#[tokio::test]
async fn write_pointer_initial_rejects_existing() {
    let dir = TempDir::new().expect("tempdir");
    let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
    let p0 = PointerFile {
        manifest_id: 0,
        manifest_list_uri: list_uri(0),
        content_hash: ContentHash([0u8; 32]),
    };
    write_pointer(&storage, &p0, None).await.expect("first");
    let err = write_pointer(&storage, &p0, None)
        .await
        .expect_err("second initial must fail");
    assert!(
        matches!(err, CommitError::WriteContentionExhausted),
        "expected WriteContentionExhausted, got {err:?}"
    );
}

#[test]
fn directory_layout_constants_match_plan() {
    assert_eq!(POINTER_PATH, "_supertable/current");
    assert_eq!(MANIFEST_LISTS_DIR, "manifest-lists");
    assert_eq!(MANIFEST_PARTS_DIR, "manifests");
    assert_eq!(
        list_uri(POINTER_ROUNDTRIP_MANIFEST_ID),
        "manifest-lists/list-000042.json"
    );
    // part_uri is hash-shaped — just sanity-check the prefix +
    // suffix.
    let h = ContentHash([0u8; 32]);
    let u = part_uri(&h);
    assert!(u.starts_with("manifests/part-"));
    assert!(u.ends_with(".avro.zst"));
}
