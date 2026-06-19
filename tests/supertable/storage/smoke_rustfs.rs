// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through a local RustFS HTTPS daemon.
//!
//! Spawns RustFS via [`infino_bench_utils::rustfs_server`], points
//! `S3StorageProvider` at it, and runs storage + supertable commit
//! probes including conditional `If-Match` PUTs.
//!
//! ## Gating
//!
//! `INFINO_TEST_RUSTFS=1` — spawning RustFS downloads or locates a
//! ~95 MiB binary on first run, so the default `cargo test` skips it.
//!
//! ```text
//! INFINO_TEST_RUSTFS=1 cargo test --test supertable storage::smoke_rustfs
//! ```

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use bytes::Bytes;
use infino::{
    supertable::{
        Supertable,
        storage::{StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use infino_bench_utils::rustfs_server;

const TEST_BUCKET: &str = "infino-rustfs-smoke";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_rustfs_https() {
    if std::env::var("INFINO_TEST_RUSTFS").is_err() {
        eprintln!(
            "supertable_smoke_via_rustfs_https: skipped (set INFINO_TEST_RUSTFS=1 to enable)"
        );
        return;
    }

    let handle = tokio::task::spawn_blocking(|| rustfs_server::spawn_rustfs(TEST_BUCKET))
        .await
        .expect("spawn_blocking join")
        .expect("spawn rustfs");
    eprintln!(
        "[rustfs-smoke] spawned on {} bucket={TEST_BUCKET}",
        handle.endpoint
    );

    let storage: Arc<dyn StorageProvider> =
        rustfs_server::rustfs_s3_provider(&handle, "").expect("rustfs provider");

    // Probe round-trip before the writer path.
    let probe_bytes = Bytes::from_static(b"hello-rustfs-smoke");
    storage
        .put_atomic("probe/hello.txt", probe_bytes.clone())
        .await
        .expect("probe put_atomic");
    let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
    assert_eq!(got, probe_bytes, "probe round-trip mismatch");

    // Conditional PUT: success with matching etag, failure with stale etag.
    storage
        .put_atomic("probe/cas.txt", Bytes::from_static(b"v1"))
        .await
        .expect("seed cas object");
    let (_, meta) = storage.get("probe/cas.txt").await.expect("read cas object");
    let etag = meta.etag.expect("etag after put_atomic");
    storage
        .put_if_match("probe/cas.txt", Bytes::from_static(b"v2"), Some(&etag))
        .await
        .expect("put_if_match with current etag");
    let stale = etag;
    let err = storage
        .put_if_match("probe/cas.txt", Bytes::from_static(b"v3"), Some(&stale))
        .await
        .expect_err("stale etag must fail");
    assert!(
        matches!(err, StorageError::PreconditionFailed { .. }),
        "expected PreconditionFailed, got {err:?}"
    );

    // Multi-commit supertable path (OCC pointer uses If-Match under the hood).
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("first commit via RustFS");
        w.append(&build_title_batch(&["echo foxtrot"]))
            .expect("second append");
        w.commit().expect("second commit via RustFS (If-Match OCC)");
        assert_eq!(producer.manifest_id(), 2);
    }

    let consumer = Supertable::open(default_supertable_options().with_storage(storage))
        .expect("open from RustFS");
    assert_eq!(consumer.manifest_id(), 2);
    assert_eq!(consumer.reader().n_docs_total(), 3);

    eprintln!("[rustfs-smoke] smoke done");
}

/// Regression: [`IngestResult`](infino_bench_utils::ingest::supertable::IngestResult) must
/// retain the RustFS child through warm/cold search. Dropping the handle while the storage
/// `Arc` is still live reproduces the connection-refused failure seen before the keepalive fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rustfs_keepalive_survives_fixture_drop() {
    if std::env::var("INFINO_TEST_RUSTFS").is_err() {
        eprintln!(
            "rustfs_keepalive_survives_fixture_drop: skipped (set INFINO_TEST_RUSTFS=1 to enable)"
        );
        return;
    }

    const PROBE_KEY: &str = "probe/keepalive.txt";
    let probe_bytes = Bytes::from_static(b"keepalive-probe");

    struct IngestLike {
        storage: Arc<dyn StorageProvider>,
        _keepalive: rustfs_server::RustFsHandle,
    }

    let fixture = {
        let handle = tokio::task::spawn_blocking(|| rustfs_server::spawn_rustfs(TEST_BUCKET))
            .await
            .expect("spawn_blocking join")
            .expect("spawn rustfs");
        let storage = rustfs_server::rustfs_s3_provider(&handle, "").expect("rustfs provider");
        storage
            .put_atomic(PROBE_KEY, probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        IngestLike {
            storage,
            _keepalive: handle,
        }
    };

    let storage = Arc::clone(&fixture.storage);
    let (got, _) = storage
        .get(PROBE_KEY)
        .await
        .expect("get with keepalive held");
    assert_eq!(
        got, probe_bytes,
        "storage must stay reachable while handle lives"
    );

    drop(fixture);
    assert!(
        storage.get(PROBE_KEY).await.is_err(),
        "dropping the RustFS handle must tear down the daemon"
    );
}
