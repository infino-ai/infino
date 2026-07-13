// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through the Azure Blob wire protocol.
//!
//! Points `AzureStorageProvider` at a running Azurite emulator (or
//! real Azure for the second test) and runs a commit + open + query
//! cycle. Every storage call (head / get / get_range / put_atomic /
//! put_if_match / delete) goes through the full Azure HTTP wire
//! protocol; nothing short-circuits to the local filesystem.
//!
//! ## Gating
//!
//! - `supertable_smoke_via_azure_wire_protocol` — `INFINO_TEST_AZURE=1`.
//!   Assumes Azurite is reachable at `http://127.0.0.1:10000`
//!   (`docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite
//!   azurite-blob --blobHost 0.0.0.0`). The test creates a fresh
//!   container per run and deletes it on success.
//! - `supertable_real_azure_round_trip` — `INFINO_TEST_REAL_AZURE=1` +
//!   `AZURE_STORAGE_CONTAINER_NAME`, with account credentials from
//!   the standard `AZURE_STORAGE_*` env chain. The container must
//!   already exist; the test scopes itself under a random prefix and
//!   cleans up after.
//!
//! Invocation:
//!
//! ```text
//! INFINO_TEST_AZURE=1 cargo test -p infino --test supertable storage::smoke_azure
//! ```

#![deny(clippy::unwrap_used)]

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use infino::{
    config::{
        CompactionSettings, Config, MemorySettings, StorageBackend, StorageColdFetchMode,
        StorageSettings, SupertableSettings,
    },
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{
        Supertable,
        manifest::disk_cache::ManifestDiskCache,
        query::VectorSearchOptions,
        storage::{AzureStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_disk_cache, default_supertable_options},
};
use tempfile::TempDir;

use super::azure_helpers::{
    EMULATOR_ENDPOINT, delete_emulator_container, ensure_emulator_container,
};

/// Substring identifying a manifest-part object URI
/// (`<prefix>/manifest-parts/part-<hex>.avro.zst`). Lets the counting
/// wrapper tell a manifest-part GET apart from pointer / manifest-list /
/// superfile GETs.
const MANIFEST_PART_URI_MARKER: &str = "manifest-parts/part-";

/// Single-thread rayon pool for deterministic Azure smoke runs.
const RAYON_POOL_THREADS: usize = 1;
/// Vector index shape for the Azure smoke fixture.
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
/// Embedding dimension for the vector smoke fixture.
const EMB_DIM: usize = 16;
/// Expected recovered doc count for the Azure round-trip fixture.
const EXPECTED_N_DOCS: u64 = 8;
/// Vector-search top-k and nprobe for the smoke ANN query.
const VECTOR_SEARCH_K: usize = 3;
const VECTOR_NPROBE: usize = 4;
/// Object / tail sizes for the `tail()` suffix-range regression test.
const TAIL_OBJECT_LEN: usize = 256;
const TAIL_FETCH_LEN: usize = 64;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn real_azure_options(dim: usize) -> infino::supertable::SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("single-thread writer pool"),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    infino::supertable::SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: VECTOR_N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: infino::superfile::vector::distance::Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8ResidualEpsilon,
        }],
        Some(infino::test_helpers::default_tokenizer()),
    )
    .expect("real Azure test options")
    .with_writer_pool(pool)
}

fn real_azure_config(container: &str, prefix: &str, cache_root: &std::path::Path) -> Config {
    Config {
        supertable: SupertableSettings::default(),
        storage: StorageSettings {
            backend: StorageBackend::Azure,
            bucket: Some(container.to_string()),
            prefix: prefix.to_string(),
            disk_cache_root: Some(cache_root.to_path_buf()),
            disk_budget_bytes: 1 << 30,
            cold_fetch_mode: StorageColdFetchMode::LazyForegroundWithBackgroundFill,
            cold_fetch_streams: 8,
            cold_fetch_chunk_bytes: 8 << 20,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            ..StorageSettings::default()
        },
        compaction: CompactionSettings::default(),
        memory: MemorySettings::default(),
    }
}

fn real_azure_batch(dim: usize) -> RecordBatch {
    let titles = LargeStringArray::from(vec![
        "alpha vector one",
        "alpha vector two",
        "bravo vector three",
        "charlie vector four",
        "delta vector five",
        "echo vector six",
        "foxtrot vector seven",
        "golf vector eight",
    ]);
    let mut flat = Vec::with_capacity(titles.len() * dim);
    for row in 0..titles.len() {
        for d in 0..dim {
            flat.push(if d == row % dim { 1.0 } else { 0.0 });
        }
    }
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(flat);
    let vectors = FixedSizeListArray::try_new(
        item_field,
        dim as i32,
        Arc::new(values) as Arc<dyn Array>,
        None,
    )
    .expect("fixed-size vector array");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(vectors)]).expect("batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_azure_wire_protocol() {
    if std::env::var("INFINO_TEST_AZURE").is_err() {
        eprintln!(
            "supertable_smoke_via_azure_wire_protocol: skipped (set INFINO_TEST_AZURE=1 to enable)"
        );
        return;
    }

    // Fresh container per run so the test is idempotent against a
    // long-lived Azurite (put_atomic is create-only and the supertable
    // pointer lives at the container root — a reused container would
    // collide on a second run).
    let container = format!("infino-azure-smoke-{}", uuid::Uuid::new_v4());
    ensure_emulator_container(&container).await;
    eprintln!("[azure] container {container} ready on {EMULATOR_ENDPOINT}");

    // Provider-level smoke first — isolates "the Azure provider works
    // at all" from "the writer + cache stack works on top".
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            AzureStorageProvider::new_with_emulator(&container).expect("azure provider for probe"),
        );
        let probe_bytes = bytes::Bytes::from_static(b"hello-azure");
        storage
            .put_atomic("probe/hello.txt", probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
        assert_eq!(got, probe_bytes, "probe round-trip mismatch");
        eprintln!("[azure] probe round-trip OK (PUT + GET via Azure wire)");
    }

    // Producer: writes through the Azure wire protocol.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            AzureStorageProvider::new_with_emulator(&container)
                .expect("azure provider for producer"),
        );
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("producer writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("producer commit via Azure");
        assert_eq!(producer.manifest_id(), 1);
        eprintln!(
            "[azure] producer commit OK; manifest_id={}",
            producer.manifest_id()
        );
    }

    // Consumer: opens via the same endpoint + a disk cache. Reads
    // route through the cache → Azure get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        AzureStorageProvider::new_with_emulator(&container).expect("azure provider for consumer"),
    );
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = default_disk_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via Azure");

    assert_eq!(consumer.manifest_id(), 1, "recovered manifest_id mismatch");
    assert_eq!(
        consumer.reader().n_docs_total(),
        2,
        "recovered n_docs_total mismatch"
    );
    eprintln!(
        "[azure] consumer open OK; manifest_id={} n_superfiles={} n_docs_total={}",
        consumer.manifest_id(),
        consumer.reader().n_superfiles(),
        consumer.reader().n_docs_total()
    );

    let pre = cache.stats();
    assert_eq!(pre.n_cold_fetches, 0);
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query_sql via Azure");
    assert_eq!(batches.len(), 1);
    let post = cache.stats();
    assert!(
        post.n_cold_fetches >= 1,
        "first query must cold-fetch through Azure; got n_cold_fetches={}",
        post.n_cold_fetches
    );
    eprintln!(
        "[azure] cold-fetch via Azure OK; n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );

    delete_emulator_container(&container).await;
    eprintln!("[azure] smoke done; container {container} deleted");
}

/// Regression: `AzureStorageProvider::tail` must not issue a suffix
/// range (`Range: bytes=-len`). object_store's Azure backend rejects
/// that with "Operation not supported: Azure does not support suffix
/// range requests", so `tail` resolves the size with a HEAD and a
/// bounded `get_range` instead. The standalone-superfile cold open
/// issues a sizeless tail to read the Parquet footer, which is what
/// surfaced this on the Azure superfile bench leg (supertable reads
/// carry `total_size` in the manifest and never reach a sizeless tail).
/// Before the fix this errored; after, it returns the trailing bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn azure_tail_uses_head_plus_range_not_suffix() {
    if std::env::var("INFINO_TEST_AZURE").is_err() {
        eprintln!(
            "azure_tail_uses_head_plus_range_not_suffix: skipped (set INFINO_TEST_AZURE=1 to enable)"
        );
        return;
    }

    let container = format!("infino-azure-tail-{}", uuid::Uuid::new_v4());
    ensure_emulator_container(&container).await;

    let storage: Arc<dyn StorageProvider> = Arc::new(
        AzureStorageProvider::new_with_emulator(&container).expect("azure provider for tail test"),
    );

    // Distinct bytes so the tail slice is unambiguous.
    let body: Vec<u8> = (0..TAIL_OBJECT_LEN).map(|i| i as u8).collect();
    storage
        .put_atomic("tail/obj.bin", bytes::Bytes::from(body.clone()))
        .await
        .expect("put tail object");

    let (tail_bytes, size) = storage
        .tail("tail/obj.bin", TAIL_FETCH_LEN as u64)
        .await
        .expect("tail must succeed on Azure (no suffix range)");
    assert_eq!(
        size, TAIL_OBJECT_LEN as u64,
        "tail must report the full object size"
    );
    assert_eq!(
        &tail_bytes[..],
        &body[TAIL_OBJECT_LEN - TAIL_FETCH_LEN..],
        "tail must return the trailing bytes"
    );

    // The `len == 0` path still resolves the size with empty bytes.
    let (empty, size_zero) = storage
        .tail("tail/obj.bin", 0)
        .await
        .expect("zero-length tail must succeed");
    assert!(empty.is_empty(), "zero-length tail returns no bytes");
    assert_eq!(size_zero, TAIL_OBJECT_LEN as u64);

    eprintln!("[azure] tail() HEAD+range OK (no suffix range)");
    delete_emulator_container(&container).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn azure_cas_conformance_holds() {
    if std::env::var("INFINO_TEST_AZURE").is_err() {
        eprintln!("azure_cas_conformance_holds: skipped (set INFINO_TEST_AZURE=1 to enable)");
        return;
    }

    let container = format!("infino-azure-cas-{}", uuid::Uuid::new_v4());
    ensure_emulator_container(&container).await;

    let storage: Arc<dyn StorageProvider> = Arc::new(
        AzureStorageProvider::new_with_emulator(&container).expect("azure provider for cas conf"),
    );
    // Azurite enforces the etag precondition, so stale rejection is asserted.
    infino::test_helpers::cas_conformance::cas_conformance(storage.as_ref(), "cas/conf", true)
        .await;

    eprintln!("[azure] CAS conformance OK");
    delete_emulator_container(&container).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_azure_conditional_get_answers_304_on_unchanged() {
    // The native `If-None-Match` path in
    // `AzureStorageProvider::get_if_none_match` — the wire-level 304
    // the refresh probe rides. Only a real Azure endpoint (or
    // Azurite) evaluates the header; the in-process s3s-fs emulator
    // does not, so this is the lane that actually proves it.
    if std::env::var("INFINO_TEST_REAL_AZURE").ok().as_deref() != Some("1") {
        eprintln!(
            "real_azure_conditional_get_answers_304_on_unchanged: skipped \
             (set INFINO_TEST_REAL_AZURE=1 and AZURE_STORAGE_CONTAINER_NAME to enable)"
        );
        return;
    }
    let container = match std::env::var("AZURE_STORAGE_CONTAINER_NAME") {
        Ok(container) => container,
        Err(_) => {
            eprintln!(
                "real_azure_conditional_get_answers_304_on_unchanged: skipped \
                 (missing AZURE_STORAGE_CONTAINER_NAME)"
            );
            return;
        }
    };
    let prefix = format!("infino-conditional-get/{}", uuid::Uuid::new_v4());
    let storage = AzureStorageProvider::new_with_prefix(
        &container,
        &prefix,
        &super::azure_helpers::azure_storage_options_from_env(),
    )
    .expect("azure provider");

    let uri = "probe/pointer";
    storage
        .put_atomic(uri, bytes::Bytes::from_static(b"v1"))
        .await
        .expect("put v1");
    let (_, meta) = storage.get(uri).await.expect("get v1");
    let etag_v1 = meta.etag.expect("azure reports etags");

    // Unchanged blob + matching etag → bodyless 304 → `None`.
    let unchanged = storage
        .get_if_none_match(uri, &etag_v1)
        .await
        .expect("conditional get against unchanged blob");
    assert!(
        unchanged.is_none(),
        "real Azure must answer If-None-Match on an unchanged blob as 304"
    );
    eprintln!("[real-azure] If-None-Match on unchanged blob → 304 OK");

    // Rewrite the blob; the stale etag must read through to the new
    // body with a fresh etag.
    storage
        .put_if_match(uri, bytes::Bytes::from_static(b"v2-longer"), Some(&etag_v1))
        .await
        .expect("cas rewrite");
    let (body, meta2) = storage
        .get_if_none_match(uri, &etag_v1)
        .await
        .expect("conditional get against rewritten blob")
        .expect("changed blob returns the body");
    assert_eq!(body, bytes::Bytes::from_static(b"v2-longer"));
    assert_ne!(meta2.etag.expect("etag"), etag_v1);
    eprintln!("[real-azure] If-None-Match on rewritten blob → 200 + new etag OK");

    storage.delete(uri).await.expect("cleanup");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn supertable_real_azure_round_trip() {
    if std::env::var("INFINO_TEST_REAL_AZURE").ok().as_deref() != Some("1") {
        eprintln!(
            "supertable_real_azure_round_trip: skipped \
             (set INFINO_TEST_REAL_AZURE=1 and AZURE_STORAGE_CONTAINER_NAME to enable)"
        );
        return;
    }

    let container = match std::env::var("AZURE_STORAGE_CONTAINER_NAME") {
        Ok(container) => container,
        Err(_) => {
            eprintln!(
                "supertable_real_azure_round_trip: skipped \
                 (missing AZURE_STORAGE_CONTAINER_NAME)"
            );
            return;
        }
    };
    let prefix_root = std::env::var("INFINO_TEST_REAL_AZURE_PREFIX")
        .unwrap_or_else(|_| "infino-real-azure-integration".to_string());
    let prefix = format!("{}/{}", prefix_root.trim_matches('/'), uuid::Uuid::new_v4());

    eprintln!("[real-azure] container={container} prefix={prefix}");

    let cache_dir = TempDir::new().expect("real Azure cache tempdir");
    let cfg = real_azure_config(&container, &prefix, cache_dir.path());
    let result = async {
        let dim = EMB_DIM;
        {
            let producer = Supertable::create(
                real_azure_options(dim)
                    .apply_config(&cfg)
                    .map_err(|e| format!("apply Azure config to producer options: {e}"))?,
            )
            .map_err(|e| format!("create unified supertable on real Azure: {e}"))?;
            let mut writer = producer
                .writer()
                .map_err(|e| format!("real Azure producer writer: {e}"))?;
            writer
                .append(&real_azure_batch(dim))
                .map_err(|e| format!("append unified vector+FTS batch: {e}"))?;
            writer
                .commit()
                .map_err(|e| format!("commit unified supertable to real Azure: {e}"))?;
            if producer.manifest_id() != 1 {
                return Err(format!(
                    "producer manifest_id mismatch: got {}",
                    producer.manifest_id()
                ));
            }
            eprintln!(
                "[real-azure] producer commit OK; manifest_id={}",
                producer.manifest_id()
            );
        }

        let consumer = Supertable::open(
            real_azure_options(dim)
                .apply_config(&cfg)
                .map_err(|e| format!("apply Azure config to consumer options: {e}"))?,
        )
        .map_err(|e| format!("open unified supertable from real Azure: {e}"))?;

        if consumer.manifest_id() != 1 {
            return Err(format!(
                "recovered manifest id mismatch: got {}",
                consumer.manifest_id()
            ));
        }
        if consumer.reader().n_docs_total() != EXPECTED_N_DOCS {
            return Err(format!(
                "recovered doc count mismatch: got {}",
                consumer.reader().n_docs_total()
            ));
        }

        let bm25_hits = consumer
            .reader()
            .bm25_search(
                "title",
                "alpha",
                10,
                infino::superfile::fts::reader::BoolMode::Or,
                None,
            )
            .map_err(|e| format!("cold BM25 over real Azure: {e}"))?;
        if bm25_hits.is_empty() {
            return Err("real Azure cold BM25 did not find alpha docs".to_string());
        }

        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let vector_hits = consumer
            .reader()
            .vector_search(
                "emb",
                &query,
                VECTOR_SEARCH_K,
                VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                None,
                None,
            )
            .map_err(|e| format!("cold vector search over real Azure: {e}"))?;
        if vector_hits.is_empty() {
            return Err("real Azure cold vector search returned no hits".to_string());
        }

        let cache = consumer
            .options()
            .disk_cache
            .as_ref()
            .ok_or_else(|| "Azure config did not attach disk cache".to_string())?;
        let stats = cache.stats();
        if stats.n_cold_fetches < 1 {
            return Err(format!(
                "real Azure reads did not hydrate through lazy disk cache; stats={stats:?}"
            ));
        }
        eprintln!(
            "[real-azure] cold lazy cache OK; n_cold_fetches={} cache_bytes={}",
            stats.n_cold_fetches, stats.current_bytes
        );

        let reader = consumer.reader();
        let manifest = reader.manifest();
        let mut cleanup_keys = vec![
            "_supertable/current".to_string(),
            infino::supertable::manifest::commit::manifest_uri(consumer.manifest_id()),
        ];
        let list_entries = manifest.get_all_list_entries();
        cleanup_keys.extend(list_entries.iter().map(|p| p.uri.clone()));
        cleanup_keys.extend(
            manifest
                .superfiles
                .iter()
                .map(|entry| entry.uri.storage_path()),
        );

        Ok::<Vec<String>, String>(cleanup_keys)
    }
    .await;

    let cleanup_storage = AzureStorageProvider::new_with_prefix(
        &container,
        &prefix,
        &super::azure_helpers::azure_storage_options_from_env(),
    )
    .expect("real Azure cleanup provider from env");
    if let Ok(keys) = &result {
        for key in keys {
            let _ = cleanup_storage.delete(key).await;
        }
    } else {
        let _ = cleanup_storage.delete("_supertable/current").await;
    }
    eprintln!("[real-azure] cleanup OK; deleted keys under prefix={prefix}");
    result.expect("real Azure integration failed");
}

/// A [`StorageProvider`] decorator that delegates to an inner provider
/// and counts GETs, separating manifest-part GETs from everything else.
/// Used to prove the manifest disk cache eliminates Azure round-trips
/// for part bytes on a warm open.
#[derive(Debug)]
struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    part_gets: AtomicUsize,
    total_gets: AtomicUsize,
}

impl CountingStorage {
    fn new(inner: Arc<dyn StorageProvider>) -> Self {
        Self {
            inner,
            part_gets: AtomicUsize::new(0),
            total_gets: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.part_gets.store(0, Ordering::Release);
        self.total_gets.store(0, Ordering::Release);
    }

    fn part_gets(&self) -> usize {
        self.part_gets.load(Ordering::Acquire)
    }

    fn note_get(&self, uri: &str) {
        self.total_gets.fetch_add(1, Ordering::AcqRel);
        if uri.contains(MANIFEST_PART_URI_MARKER) {
            self.part_gets.fetch_add(1, Ordering::AcqRel);
        }
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(bytes::Bytes, ObjectMeta), StorageError> {
        self.note_get(uri);
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<bytes::Bytes, StorageError> {
        // Manifest parts load via `get` (full object); count any
        // part-range GETs too so the warm-path assertion stays honest.
        self.note_get(uri);
        self.inner.get_range(uri, range).await
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(bytes::Bytes, u64), StorageError> {
        self.inner.tail(uri, len).await
    }

    async fn put_atomic(
        &self,
        uri: &str,
        bytes: bytes::Bytes,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: bytes::Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.inner.list_with_prefix_metadata(prefix).await
    }
}

/// Real-Azure validation of the manifest-part disk cache.
///
/// 1. A producer commits a batch to Azure (creates ≥1 manifest part).
/// 2. A cold consumer opens with a fresh [`ManifestDiskCache`] — the
///    eager part load fetches the part from Azure (≥1 part GET) and
///    populates the cache.
/// 3. A warm consumer opens with a **new** `ManifestDiskCache` instance
///    over the **same** cache directory (models a process restart): the
///    index is rebuilt from disk, the eager load is served from local
///    disk, and **zero** Azure part GETs occur.
///
/// Gated on `INFINO_TEST_REAL_AZURE=1` + `AZURE_STORAGE_CONTAINER_NAME`
/// (with the standard `AZURE_STORAGE_*` credential env chain).
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn manifest_disk_cache_serves_parts_without_azure_refetch() {
    if std::env::var("INFINO_TEST_REAL_AZURE").ok().as_deref() != Some("1") {
        eprintln!(
            "manifest_disk_cache_serves_parts_without_azure_refetch: skipped \
             (set INFINO_TEST_REAL_AZURE=1 and AZURE_STORAGE_CONTAINER_NAME to enable)"
        );
        return;
    }
    let container = match std::env::var("AZURE_STORAGE_CONTAINER_NAME") {
        Ok(c) => c,
        Err(_) => {
            eprintln!(
                "manifest_disk_cache_serves_parts_without_azure_refetch: skipped \
                 (missing AZURE_STORAGE_CONTAINER_NAME)"
            );
            return;
        }
    };
    let prefix_root = std::env::var("INFINO_TEST_REAL_AZURE_PREFIX")
        .unwrap_or_else(|_| "infino-real-azure-manifest-cache".to_string());
    let prefix = format!("{}/{}", prefix_root.trim_matches('/'), uuid::Uuid::new_v4());
    eprintln!("[real-azure mcache] container={container} prefix={prefix}");

    let dim = EMB_DIM;
    let cache_dir = TempDir::new().expect("manifest-cache tempdir");
    // The manifest cache's own subdirectory; reused across both opens so
    // the warm open's fresh instance scans the cold open's files.
    let manifest_cache_root = cache_dir.path().join("manifest-parts");

    let result = async {
        let azure: Arc<dyn StorageProvider> = Arc::new(
            AzureStorageProvider::new_with_prefix(
                &container,
                &prefix,
                &super::azure_helpers::azure_storage_options_from_env(),
            )
            .map_err(|e| format!("azure provider: {e}"))?,
        );
        let counting = Arc::new(CountingStorage::new(Arc::clone(&azure)));
        let counting_dyn: Arc<dyn StorageProvider> =
            Arc::clone(&counting) as Arc<dyn StorageProvider>;

        // 1. Producer commit → ≥1 manifest part on Azure.
        {
            let producer =
                Supertable::create(real_azure_options(dim).with_storage(Arc::clone(&counting_dyn)))
                    .map_err(|e| format!("create producer: {e}"))?;
            let mut writer = producer
                .writer()
                .map_err(|e| format!("producer writer: {e}"))?;
            writer
                .append(&real_azure_batch(dim))
                .map_err(|e| format!("append: {e}"))?;
            writer.commit().map_err(|e| format!("commit: {e}"))?;
            if producer.manifest_id() != 1 {
                return Err(format!("producer manifest_id={}", producer.manifest_id()));
            }
            eprintln!("[real-azure mcache] producer commit OK; manifest_id=1");
        }

        // 2. Cold open: fresh manifest cache → part fetched from Azure.
        let cold_cache_dir = cache_dir.path().join("cold-superfiles");
        {
            let manifest_cache = ManifestDiskCache::new(manifest_cache_root.clone(), 1 << 30)
                .map_err(|e| format!("cold manifest cache: {e}"))?;
            let disk_cache = default_disk_cache(Arc::clone(&counting_dyn), &cold_cache_dir);
            counting.reset();
            let consumer = Supertable::open(
                real_azure_options(dim)
                    .with_storage(Arc::clone(&counting_dyn))
                    .with_disk_cache(disk_cache)
                    .with_manifest_disk_cache(Arc::clone(&manifest_cache)),
            )
            .map_err(|e| format!("cold open: {e}"))?;

            if consumer.reader().n_docs_total() != EXPECTED_N_DOCS {
                return Err(format!(
                    "cold n_docs_total mismatch: {}",
                    consumer.reader().n_docs_total()
                ));
            }
            let cold_part_gets = counting.part_gets();
            let stats = manifest_cache.stats();
            if cold_part_gets < 1 {
                return Err(format!(
                    "cold open should fetch ≥1 part from Azure; part_gets={cold_part_gets}"
                ));
            }
            if stats.n_entries < 1 {
                return Err(format!(
                    "cold open did not populate manifest cache; {stats:?}"
                ));
            }
            eprintln!(
                "[real-azure mcache] cold open OK; azure_part_gets={cold_part_gets} \
                 cache_entries={} cache_misses={}",
                stats.n_entries, stats.n_misses
            );
        }

        // 3. Warm open: NEW cache instance over the same dir (restart).
        let warm_cache_dir = cache_dir.path().join("warm-superfiles");
        {
            let manifest_cache = ManifestDiskCache::new(manifest_cache_root.clone(), 1 << 30)
                .map_err(|e| format!("warm manifest cache: {e}"))?;
            // Restart survival: the index is rebuilt from the cold open's
            // files at construction, before any load.
            let scanned = manifest_cache.stats();
            if scanned.n_entries < 1 {
                return Err(format!(
                    "warm cache did not rebuild index from disk; {scanned:?}"
                ));
            }

            let disk_cache = default_disk_cache(Arc::clone(&counting_dyn), &warm_cache_dir);
            counting.reset();
            let consumer = Supertable::open(
                real_azure_options(dim)
                    .with_storage(Arc::clone(&counting_dyn))
                    .with_disk_cache(disk_cache)
                    .with_manifest_disk_cache(Arc::clone(&manifest_cache)),
            )
            .map_err(|e| format!("warm open: {e}"))?;

            if consumer.reader().n_docs_total() != EXPECTED_N_DOCS {
                return Err(format!(
                    "warm n_docs_total mismatch: {}",
                    consumer.reader().n_docs_total()
                ));
            }
            let warm_part_gets = counting.part_gets();
            let stats = manifest_cache.stats();
            if warm_part_gets != 0 {
                return Err(format!(
                    "warm open must serve parts from disk cache; \
                     got {warm_part_gets} Azure part GETs (expected 0)"
                ));
            }
            if stats.n_hits < 1 {
                return Err(format!("warm open recorded no cache hit; {stats:?}"));
            }
            eprintln!(
                "[real-azure mcache] warm open OK; azure_part_gets=0 \
                 cache_hits={} (parts served from disk)",
                stats.n_hits
            );

            // Collect cleanup keys from the warm consumer's manifest.
            let reader = consumer.reader();
            let manifest = reader.manifest();
            let mut keys = vec![
                "_supertable/current".to_string(),
                infino::supertable::manifest::commit::manifest_uri(consumer.manifest_id()),
            ];
            keys.extend(
                manifest
                    .get_all_list_entries()
                    .iter()
                    .map(|p| p.uri.clone()),
            );
            keys.extend(manifest.superfiles.iter().map(|e| e.uri.storage_path()));
            Ok::<Vec<String>, String>(keys)
        }
    }
    .await;

    let cleanup = AzureStorageProvider::new_with_prefix(
        &container,
        &prefix,
        &super::azure_helpers::azure_storage_options_from_env(),
    )
    .expect("real Azure cleanup provider");
    match &result {
        Ok(keys) => {
            for key in keys {
                let _ = cleanup.delete(key).await;
            }
        }
        Err(_) => {
            let _ = cleanup.delete("_supertable/current").await;
        }
    }
    eprintln!("[real-azure mcache] cleanup done under prefix={prefix}");
    result.expect("real Azure manifest-cache test failed");
}
