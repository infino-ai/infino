// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through the GCS wire protocol (fake-gcs-server).
//!
//! Gated on `INFINO_TEST_GCS=1`. Every storage call (head / get /
//! get_range / put_atomic / put_if_match / delete / list) rides the GCS
//! HTTP wire; nothing short-circuits to the local filesystem. The
//! `cas_conformance` step verifies the generation-keyed conditional-write
//! path end to end — the commit pointer CAS depends on it.
//!
//! Invocation:
//!   docker run -d --rm -p 4443:4443 fsouza/fake-gcs-server \
//!     -scheme http -public-host 127.0.0.1:4443
//!   INFINO_TEST_GCS=1 cargo test -p infino --test supertable storage::smoke_gcs

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use infino::{
    supertable::{
        Supertable,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{GcsStorageProvider, StorageProvider},
    },
    test_helpers::{
        build_title_batch, cas_conformance::cas_conformance, default_supertable_options,
    },
};
use tempfile::TempDir;

use super::gcs_helpers::{EMULATOR_ENDPOINT, delete_emulator_bucket, ensure_emulator_bucket};

/// Disk-cache byte budget for the consumer (1 GiB; the fixture is tiny).
const CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Cold-fetch stream fan-out and chunk size for the smoke consumer.
const COLD_FETCH_STREAMS: usize = 4;
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;

fn make_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

fn gcs_enabled() -> bool {
    std::env::var("INFINO_TEST_GCS").is_ok_and(|v| v == "1")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_gcs_wire_protocol() {
    if !gcs_enabled() {
        eprintln!("supertable_smoke_via_gcs_wire_protocol: skipped (set INFINO_TEST_GCS=1)");
        return;
    }

    // Fresh bucket per run so the create-only pointer PUT doesn't collide
    // with a prior run against a long-lived emulator.
    let bucket = format!("infino-gcs-smoke-{}", uuid::Uuid::new_v4());
    ensure_emulator_bucket(&bucket).await;
    eprintln!("[gcs] bucket {bucket} ready on {EMULATOR_ENDPOINT}");

    // Provider-level smoke: probe round-trip + full CAS conformance
    // (generation-keyed; fake-gcs-server enforces if-generation-match).
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            GcsStorageProvider::new_with_emulator(EMULATOR_ENDPOINT, &bucket)
                .expect("gcs provider for probe"),
        );
        let probe = bytes::Bytes::from_static(b"hello-gcs");
        storage
            .put_atomic("probe/hello.txt", probe.clone())
            .await
            .expect("probe put_atomic");
        let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
        assert_eq!(got, probe, "probe round-trip mismatch");

        cas_conformance(storage.as_ref(), "cas/conf", true).await;
        eprintln!("[gcs] probe round-trip + CAS conformance OK");
    }

    // Producer: writes + commits through the GCS wire.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            GcsStorageProvider::new_with_emulator(EMULATOR_ENDPOINT, &bucket)
                .expect("gcs provider for producer"),
        );
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("producer writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("producer commit via GCS");
        assert_eq!(producer.manifest_id(), 1);
        eprintln!(
            "[gcs] producer commit OK; manifest_id={}",
            producer.manifest_id()
        );
    }

    // Consumer: opens via the same endpoint + a disk cache; reads route
    // through the cache → GCS get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        GcsStorageProvider::new_with_emulator(EMULATOR_ENDPOINT, &bucket)
            .expect("gcs provider for consumer"),
    );
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = make_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via GCS");

    assert_eq!(consumer.manifest_id(), 1, "recovered manifest_id mismatch");
    assert_eq!(
        consumer.reader().n_docs_total(),
        2,
        "recovered n_docs_total mismatch"
    );

    let pre = cache.stats();
    assert_eq!(pre.n_cold_fetches, 0);
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query_sql via GCS");
    assert_eq!(batches.len(), 1);
    let post = cache.stats();
    assert!(
        post.n_cold_fetches >= 1,
        "first query must cold-fetch through GCS; got n_cold_fetches={}",
        post.n_cold_fetches
    );
    eprintln!(
        "[gcs] cold-fetch via GCS OK; n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );

    delete_emulator_bucket(&bucket).await;
    eprintln!("[gcs] smoke done; bucket {bucket} deleted");
}
