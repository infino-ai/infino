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

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::config::{
    CompactionSettings, Config, StorageBackend, StorageColdFetchMode, StorageSettings,
    SupertableSettings,
};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::supertable::Supertable;
use infino::supertable::query::VectorSearchOptions;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{AzureStorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options};
use tempfile::TempDir;

use super::azure_helpers::{
    EMULATOR_ENDPOINT, delete_emulator_container, ensure_emulator_container,
};

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

fn make_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: 1 << 30,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: 4,
        cold_fetch_chunk_bytes: 1 << 20,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

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
    let cache = make_cache(Arc::clone(&consumer_storage), cache_dir.path());

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
            infino::supertable::manifest::commit::list_uri(consumer.manifest_id()),
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

    let cleanup_storage = AzureStorageProvider::new_with_prefix(&container, &prefix)
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
