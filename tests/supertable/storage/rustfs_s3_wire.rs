// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `S3StorageProvider` wire round-trips over local RustFS HTTPS.
//!
//! Granular `StorageProvider` method tests (formerly in `src/storage/s3.rs`).
//! Uses the lazy shared session via [`rustfs_server::open_test_fixture`].
//!
//! Runs by default. Set `INFINO_TEST_DISABLE_RUSTFS=1` to skip.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use bytes::Bytes;
use infino::supertable::storage::{StorageError, StorageProvider};
use infino_bench_utils::rustfs_server;

async fn harness_storage() -> Arc<dyn StorageProvider> {
    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open test fixture");
    Arc::clone(&fixture.storage)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_atomic_then_get_round_trips() {
    if !rustfs_server::begin_rustfs_test("put_atomic_then_get_round_trips") {
        return;
    }
    let p = harness_storage().await;
    let body = Bytes::from_static(b"hello-unit-s3");
    p.put_atomic("k/hello.txt", body.clone())
        .await
        .expect("put_atomic");
    let (got, meta) = p.get("k/hello.txt").await.expect("get");
    assert_eq!(got, body);
    assert_eq!(meta.size, body.len() as u64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_atomic_twice_is_precondition_failed() {
    if !rustfs_server::begin_rustfs_test("put_atomic_twice_is_precondition_failed") {
        return;
    }
    let p = harness_storage().await;
    let body = Bytes::from_static(b"first");
    p.put_atomic("k/dup", body.clone())
        .await
        .expect("first put");
    let err = p
        .put_atomic("k/dup", Bytes::from_static(b"second"))
        .await
        .expect_err("second create must fail");
    assert!(
        matches!(err, StorageError::PreconditionFailed { .. }),
        "expected PreconditionFailed; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_missing_is_not_found() {
    if !rustfs_server::begin_rustfs_test("get_missing_is_not_found") {
        return;
    }
    let p = harness_storage().await;
    let err = p.get("k/absent").await.expect_err("get missing must fail");
    assert!(
        matches!(err, StorageError::NotFound { .. }),
        "expected NotFound; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn head_missing_is_not_found() {
    if !rustfs_server::begin_rustfs_test("head_missing_is_not_found") {
        return;
    }
    let p = harness_storage().await;
    let err = p.head("k/absent").await.expect_err("head missing fails");
    assert!(
        matches!(err, StorageError::NotFound { .. }),
        "expected NotFound; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn head_reports_size() {
    if !rustfs_server::begin_rustfs_test("head_reports_size") {
        return;
    }
    let p = harness_storage().await;
    let body = Bytes::from_static(b"0123456789");
    p.put_atomic("k/sized", body.clone())
        .await
        .expect("put_atomic");
    let meta = p.head("k/sized").await.expect("head");
    assert_eq!(meta.size, body.len() as u64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_range_returns_subslice() {
    if !rustfs_server::begin_rustfs_test("get_range_returns_subslice") {
        return;
    }
    let p = harness_storage().await;
    let body: Vec<u8> = (0..=255u8).collect();
    p.put_atomic("k/range.bin", Bytes::from(body.clone()))
        .await
        .expect("put_atomic");
    let got = p.get_range("k/range.bin", 10..20).await.expect("get_range");
    assert_eq!(&got[..], &body[10..20]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tail_returns_trailing_bytes_and_size() {
    if !rustfs_server::begin_rustfs_test("tail_returns_trailing_bytes_and_size") {
        return;
    }
    let p = harness_storage().await;
    let body: Vec<u8> = (0..200u8).collect();
    p.put_atomic("k/tail.bin", Bytes::from(body.clone()))
        .await
        .expect("put_atomic");
    let (tail, size) = p.tail("k/tail.bin", 32).await.expect("tail");
    assert_eq!(size, body.len() as u64);
    assert_eq!(&tail[..], &body[body.len() - 32..]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tail_zero_len_falls_back_to_head_for_size() {
    if !rustfs_server::begin_rustfs_test("tail_zero_len_falls_back_to_head_for_size") {
        return;
    }
    let p = harness_storage().await;
    let body = Bytes::from_static(b"abcdef");
    p.put_atomic("k/tail0.bin", body.clone())
        .await
        .expect("put_atomic");
    let (tail, size) = p.tail("k/tail0.bin", 0).await.expect("zero-len tail");
    assert!(tail.is_empty());
    assert_eq!(size, body.len() as u64);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_removes_object() {
    if !rustfs_server::begin_rustfs_test("delete_removes_object") {
        return;
    }
    let p = harness_storage().await;
    p.put_atomic("k/del", Bytes::from_static(b"x"))
        .await
        .expect("put_atomic");
    p.delete("k/del").await.expect("delete existing");
    let err = p.get("k/del").await.expect_err("deleted object gone");
    assert!(matches!(err, StorageError::NotFound { .. }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_with_prefix_returns_matching_keys() {
    if !rustfs_server::begin_rustfs_test("list_with_prefix_returns_matching_keys") {
        return;
    }
    let p = harness_storage().await;
    p.put_atomic("list/a.txt", Bytes::from_static(b"a"))
        .await
        .expect("put a");
    p.put_atomic("list/b.txt", Bytes::from_static(b"b"))
        .await
        .expect("put b");
    p.put_atomic("other/c.txt", Bytes::from_static(b"c"))
        .await
        .expect("put c");
    let mut keys = p.list_with_prefix("list/").await.expect("list");
    keys.sort();
    assert_eq!(
        keys,
        vec!["list/a.txt".to_string(), "list/b.txt".to_string()]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_if_match_none_is_create_only() {
    if !rustfs_server::begin_rustfs_test("put_if_match_none_is_create_only") {
        return;
    }
    let p = harness_storage().await;
    p.put_if_match("k/cas", Bytes::from_static(b"v1"), None)
        .await
        .expect("create-if-absent");
    let err = p
        .put_if_match("k/cas", Bytes::from_static(b"v2"), None)
        .await
        .expect_err("second create-if-absent must fail");
    assert!(
        matches!(err, StorageError::PreconditionFailed { .. }),
        "expected PreconditionFailed; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn put_if_match_etag_update_succeeds_with_matching_etag() {
    if !rustfs_server::begin_rustfs_test("put_if_match_etag_update_succeeds_with_matching_etag") {
        return;
    }
    let p = harness_storage().await;
    let etag = p
        .put_atomic("k/upd", Bytes::from_static(b"v1"))
        .await
        .expect("initial put")
        .expect("rustfs returns an etag on create");
    p.put_if_match("k/upd", Bytes::from_static(b"v2"), Some(&etag))
        .await
        .expect("update with matching etag");
    let (got, _) = p.get("k/upd").await.expect("get latest");
    assert_eq!(&got[..], b"v2");
}
