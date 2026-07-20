// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable end-to-end round-trip over the Google Cloud Storage wire.
//!
//! Gated on `INFINO_TEST_REAL_GCS=1` plus `INFINO_GCS_BUCKET` and
//! `GOOGLE_APPLICATION_CREDENTIALS` (a service-account key path). Every
//! storage call rides the real GCS HTTP wire; nothing short-circuits to the
//! local filesystem. Mirrors `supertable_real_azure_round_trip`: byte-level
//! generation CAS (`cas_conformance`), then a unified FTS + vector commit →
//! reopen → BM25 + vector + SQL query cycle through a lazy disk cache,
//! deleting every object it wrote under its unique prefix.
//!
//! No emulator variant: the common GCS emulators don't faithfully implement
//! the XML write API `object_store` uses (fake-gcs-server has no XML PUT;
//! storage-testbench's XML PUT omits the `ETag`/`x-goog-generation` response
//! headers `object_store` requires), so real GCS is the write-path gate.
//!
//! Invocation:
//!   INFINO_TEST_REAL_GCS=1 INFINO_GCS_BUCKET=<bucket> \
//!   GOOGLE_APPLICATION_CREDENTIALS=<sa-key.json> \
//!   cargo test -p infino --test supertable storage::smoke_gcs -- --nocapture

#![deny(clippy::unwrap_used)]

use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::{
        builder::{FtsConfig, VectorConfig},
        fts::reader::BoolMode,
        vector::{distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{
        Supertable, SupertableOptions,
        query::VectorSearchOptions,
        storage::{GcsStorageProvider, StorageProvider},
    },
    test_helpers::{cas_conformance::cas_conformance, default_disk_cache, default_tokenizer},
};
use tempfile::TempDir;

/// Vector-index shape for the unified fixture.
const EMB_DIM: usize = 16;
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
/// Docs in the fixture (one-hot embeddings, distinct titles).
const EXPECTED_N_DOCS: u64 = 8;
/// Vector-search top-k and nprobe for the ANN query.
const VECTOR_SEARCH_K: usize = 3;
const VECTOR_NPROBE: usize = 4;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn unified_schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]))
}

/// FTS(title) + vector(emb) options, single-thread writer pool for determinism.
fn gcs_options(dim: usize) -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread writer pool"),
    );
    SupertableOptions::new(
        unified_schema(dim),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: VECTOR_N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        }],
        Some(default_tokenizer()),
    )
    .expect("gcs test options")
    .with_writer_pool(pool)
}

/// Eight docs: distinct titles + one-hot embeddings (row `r` → basis vector `r % dim`).
fn gcs_batch(dim: usize) -> RecordBatch {
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
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
        None,
    )
    .expect("fixed-size vector array");
    RecordBatch::try_new(
        unified_schema(dim),
        vec![Arc::new(titles), Arc::new(vectors)],
    )
    .expect("batch")
}

/// Real-GCS config from env: `(bucket, unique_prefix, sa_key_path)`. `None`
/// unless both `INFINO_GCS_BUCKET` and `GOOGLE_APPLICATION_CREDENTIALS` (a
/// service-account key path) are set. The prefix carries a per-run UUID so
/// concurrent/repeat runs never collide and cleanup stays scoped.
fn real_gcs_env() -> Option<(String, String, String)> {
    let bucket = std::env::var("INFINO_GCS_BUCKET").ok()?;
    let key_path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok()?;
    let root = std::env::var("INFINO_TEST_REAL_GCS_PREFIX")
        .unwrap_or_else(|_| "infino-real-gcs-integration".to_string());
    let prefix = format!("{}/{}", root.trim_matches('/'), uuid::Uuid::new_v4());
    Some((bucket, prefix, key_path))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_real_gcs_round_trip() {
    if std::env::var("INFINO_TEST_REAL_GCS").ok().as_deref() != Some("1") {
        eprintln!(
            "supertable_real_gcs_round_trip: skipped (set INFINO_TEST_REAL_GCS=1 + \
             INFINO_GCS_BUCKET + GOOGLE_APPLICATION_CREDENTIALS)"
        );
        return;
    }
    let Some((bucket, prefix, key_path)) = real_gcs_env() else {
        eprintln!(
            "supertable_real_gcs_round_trip: skipped (missing INFINO_GCS_BUCKET or \
             GOOGLE_APPLICATION_CREDENTIALS)"
        );
        return;
    };
    eprintln!("[real-gcs] bucket={bucket} prefix={prefix}");
    let opts = HashMap::from([("google_service_account".to_string(), key_path)]);

    // Prefix-scoped provider: every object this run writes lands under `prefix`.
    let storage: Arc<dyn StorageProvider> = Arc::new(
        GcsStorageProvider::new_with_prefix(&bucket, &prefix, &opts).expect("real gcs provider"),
    );

    // 1. Byte-level CAS conformance over the real GCS wire (generation-keyed;
    //    real GCS enforces if-generation-match, so stale rejection is asserted).
    cas_conformance(storage.as_ref(), "cas/conf", true).await;
    eprintln!("[real-gcs] CAS conformance OK");

    // 2. Unified FTS + vector commit through the real GCS wire.
    {
        let producer = Supertable::create(gcs_options(EMB_DIM).with_storage(Arc::clone(&storage)))
            .expect("create real gcs supertable");
        let mut w = producer.writer().expect("writer");
        w.append(&gcs_batch(EMB_DIM)).expect("append");
        w.commit().expect("commit to real gcs");
        assert_eq!(producer.manifest_id(), 1);
    }

    // 3. Reopen with a lazy disk cache; BM25 + vector + SQL all read through GCS.
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        gcs_options(EMB_DIM)
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open real gcs supertable");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), EXPECTED_N_DOCS);

    let bm25 = consumer
        .reader()
        .bm25_search("title", "alpha", 10, BoolMode::Or, None)
        .expect("bm25 over real gcs");
    assert!(!bm25.is_empty(), "cold BM25 must find the alpha docs");

    let mut query = vec![0.0f32; EMB_DIM];
    query[0] = 1.0;
    let vectors = consumer
        .reader()
        .vector_search(
            "emb",
            &query,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("vector search over real gcs");
    assert!(!vectors.is_empty(), "cold vector search must return hits");

    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query real gcs");
    assert_eq!(batches.len(), 1);
    assert!(
        cache.stats().n_cold_fetches >= 1,
        "reads must cold-fetch through GCS"
    );
    eprintln!(
        "[real-gcs] commit+reopen+bm25+vector+sql OK; n_cold_fetches={}",
        cache.stats().n_cold_fetches
    );

    // 4. Cleanup: a non-prefixed provider lists by absolute key and deletes
    //    every object under our unique prefix (list is absolute, delete on an
    //    empty-prefix provider is absolute — no double-prefixing).
    let cleanup: Arc<dyn StorageProvider> = Arc::new(
        GcsStorageProvider::new_with_prefix(&bucket, "", &opts).expect("cleanup provider"),
    );
    let keys = cleanup
        .list_with_prefix(&prefix)
        .await
        .expect("list cleanup");
    for key in &keys {
        cleanup.delete(key).await.expect("cleanup delete");
    }
    eprintln!(
        "[real-gcs] cleaned up {} objects under {prefix}",
        keys.len()
    );
}
