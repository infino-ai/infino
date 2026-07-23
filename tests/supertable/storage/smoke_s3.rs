// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable integration against **real AWS S3**.
//!
//! Local S3 wire-protocol smoke (RustFS HTTPS daemon, conditional PUTs,
//! search TVFs) lives in [`smoke_rustfs`]. This module exercises production
//! S3 credentials and a live bucket only.
//!
//! ## Gating
//!
//! `INFINO_TEST_REAL_S3=1` plus `INFINO_TEST_REAL_S3_BUCKET` (or
//! `INFINO_TEST_S3_BUCKET`). Without them the test exits as a no-op.
//!
//! ```text
//! INFINO_TEST_REAL_S3=1 INFINO_TEST_REAL_S3_BUCKET=my-bucket \
//!   cargo test --test supertable storage::smoke_s3
//! ```

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    config::{
        CompactionSettings, Config, StorageBackend, StorageColdFetchMode, StorageSettings,
        SupertableSettings,
    },
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{
        Supertable,
        query::VectorSearchOptions,
        storage::{S3StorageProvider, StorageProvider},
    },
};
use tempfile::TempDir;

/// Single-thread rayon pool for deterministic S3 smoke runs.
const RAYON_POOL_THREADS: usize = 1;
/// Vector index shape for the S3 smoke fixture.
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
/// Embedding dimension for the vector smoke fixtures.
const EMB_DIM: usize = 16;
/// Expected recovered doc count for the S3 round-trip fixtures.
const EXPECTED_N_DOCS: u64 = 8;
/// Vector-search top-k and nprobe for the smoke ANN query.
const VECTOR_SEARCH_K: usize = 3;
const VECTOR_NPROBE: usize = 4;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn real_s3_options(dim: usize) -> infino::supertable::SupertableOptions {
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
            positions: false,
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: VECTOR_N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: infino::superfile::vector::distance::Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
            provided_centroids: None,
        }],
        Some(infino::test_helpers::default_tokenizer()),
    )
    .expect("real S3 test options")
    .with_writer_pool(pool)
}

/// Real-S3 credential options from the AWS environment, for the gated
/// `INFINO_TEST_REAL_S3` test. Infino's provider no longer reads the
/// environment; the test passes these as config.
fn s3_storage_options_from_env() -> std::collections::HashMap<String, String> {
    // AWS_DEFAULT_REGION before AWS_REGION so the latter wins when both
    // are set (equal keys, last insert wins).
    [
        ("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
        ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
        ("AWS_SESSION_TOKEN", "aws_session_token"),
        ("AWS_DEFAULT_REGION", "aws_region"),
        ("AWS_REGION", "aws_region"),
    ]
    .iter()
    .filter_map(|(env, key)| std::env::var(env).ok().map(|v| (key.to_string(), v)))
    .collect()
}

fn real_s3_config(bucket: &str, prefix: &str, cache_root: &std::path::Path) -> Config {
    Config {
        supertable: SupertableSettings::default(),
        storage: StorageSettings {
            backend: StorageBackend::S3,
            bucket: Some(bucket.to_string()),
            storage_options: s3_storage_options_from_env(),
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
        ..Config::default()
    }
}

fn real_s3_batch(dim: usize) -> RecordBatch {
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
    let vectors = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
        .expect("vectors");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(vectors)]).expect("batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn supertable_real_s3_lazy_vector_and_fts_round_trip() {
    if std::env::var("INFINO_TEST_REAL_S3").ok().as_deref() != Some("1") {
        eprintln!(
            "supertable_real_s3_lazy_vector_and_fts_round_trip: skipped \
             (set INFINO_TEST_REAL_S3=1 and INFINO_TEST_REAL_S3_BUCKET to enable)"
        );
        return;
    }

    let bucket = match std::env::var("INFINO_TEST_REAL_S3_BUCKET")
        .or_else(|_| std::env::var("INFINO_TEST_S3_BUCKET"))
    {
        Ok(bucket) => bucket,
        Err(_) => {
            eprintln!(
                "supertable_real_s3_lazy_vector_and_fts_round_trip: skipped \
                 (missing INFINO_TEST_REAL_S3_BUCKET)"
            );
            return;
        }
    };
    let prefix_root = std::env::var("INFINO_TEST_REAL_S3_PREFIX")
        .unwrap_or_else(|_| "infino-real-s3-integration".to_string());
    let prefix = format!("{}/{}", prefix_root.trim_matches('/'), uuid::Uuid::new_v4());

    eprintln!("[real-s3] bucket={bucket} prefix={prefix}");

    let storage_opts = s3_storage_options_from_env();
    // Do not require explicit AWS_* env vars: the provider can also resolve the
    // default credential chain (instance role, shared config, etc.).
    eprintln!(
        "[real-s3] storage_options keys: {:?}",
        storage_opts.keys().collect::<Vec<_>>()
    );

    let cache_dir = TempDir::new().expect("real S3 cache tempdir");
    let cfg = real_s3_config(&bucket, &prefix, cache_dir.path());
    let result = async {
        let dim = EMB_DIM;
        {
            let producer = Supertable::create(
                real_s3_options(dim)
                    .apply_config(&cfg)
                    .map_err(|e| format!("apply S3 config to producer options: {e}"))?,
            )
            .map_err(|e| format!("create unified supertable on real S3: {e}"))?;
            let mut writer = producer
                .writer()
                .map_err(|e| format!("real S3 producer writer: {e}"))?;
            writer
                .append(&real_s3_batch(dim))
                .map_err(|e| format!("append unified vector+FTS batch: {e}"))?;
            writer
                .commit()
                .map_err(|e| format!("commit unified supertable to real S3: {e}"))?;
            if producer.manifest_id() != 1 {
                return Err(format!(
                    "producer manifest_id mismatch: got {}",
                    producer.manifest_id()
                ));
            }
            eprintln!(
                "[real-s3] producer commit OK; manifest_id={}",
                producer.manifest_id()
            );
        }

        let consumer = Supertable::open(
            real_s3_options(dim)
                .apply_config(&cfg)
                .map_err(|e| format!("apply S3 config to consumer options: {e}"))?,
        )
        .map_err(|e| format!("open unified supertable from real S3: {e}"))?;

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
            .map_err(|e| format!("cold BM25 over real S3: {e}"))?;
        if bm25_hits.is_empty() {
            return Err("real S3 cold BM25 did not find alpha docs".to_string());
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
            .map_err(|e| format!("cold vector search over real S3: {e}"))?;
        if vector_hits.is_empty() {
            return Err("real S3 cold vector search returned no hits".to_string());
        }

        let cache = consumer
            .options()
            .disk_cache
            .as_ref()
            .ok_or_else(|| "S3 config did not attach disk cache".to_string())?;
        let stats = cache.stats();
        if stats.n_cold_fetches < 1 {
            return Err(format!(
                "real S3 reads did not hydrate through lazy disk cache; stats={stats:?}"
            ));
        }
        eprintln!(
            "[real-s3] cold lazy cache OK; n_cold_fetches={} cache_bytes={}",
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
    let cleanup_storage =
        S3StorageProvider::new_with_prefix(&bucket, &prefix, &s3_storage_options_from_env())
            .expect("real S3 cleanup provider from AWS env");
    if let Ok(keys) = &result {
        for key in keys {
            let _ = cleanup_storage.delete(key).await;
        }
    } else {
        let _ = cleanup_storage.delete("_supertable/current").await;
    }
    eprintln!("[real-s3] cleanup OK; deleted keys under prefix={prefix}");
    if let Err(ref err) = result {
        eprintln!("[real-s3] error detail: {err}");
    }
    result.expect("real S3 integration failed");
}
