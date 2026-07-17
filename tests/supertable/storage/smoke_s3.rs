// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through the S3 wire protocol.
//!
//! Stands up an in-process s3s-fs server on a random port,
//! points `S3StorageProvider` at it, and runs a small
//! commit + open + query cycle. Validates the "real cloud
//! path" end-to-end: every storage call (head / get /
//! get_range / put_atomic / put_if_match / delete) goes
//! through the full S3 HTTP wire protocol; nothing
//! short-circuits to the local filesystem.
//!
//! ## Gating
//!
//! The test is gated on `INFINO_TEST_S3=1`. Without the env
//! var, the test exits as a no-op early (printing a brief
//! "skipped" line). Reason: spawning an in-process HTTP
//! server has cost (~50 ms per test invocation) and pulls
//! in s3s + s3s-fs dev-dependencies on the test binary's
//! compile path. The default `cargo test` run skips it.
//!
//! Invocation:
//!
//! ```text
//! INFINO_TEST_S3=1 cargo test --test supertable_smoke_s3
//! ```
//!
//! ## What's verified
//!
//! - `Supertable::create + writer.commit` against the S3
//!   wire path (superfiles + manifest part + manifest +
//!   pointer all PUT via HTTP).
//! - `Supertable::open` from a fresh handle recovers the
//!   pre-commit state (manifest_id, n_superfiles, n_docs_total).
//! - Reader query via `query_sql` routes through the
//!   `DiskCacheStore` (cold-fetch via HTTP get_range from
//!   the s3s-fs server).
//!
//! ## What's NOT verified
//!
//! - AWS-specific quirks: virtual-hosted-style requests,
//!   AWS-Sig-V4 authentication corner cases, regional
//!   endpoints. The smoke test uses path-style (forced) +
//!   a fixed dummy credential pair. Real-AWS validation
//!   requires AWS credentials + a test bucket; out of scope
//!   for an in-process smoke.
//! - Concurrent writers (the OCC retry is exercised
//!   end-to-end in `tests/supertable_concurrent_processes.rs`
//!   against LocalFS; the S3 path uses S3 CAS natively, no
//!   read-then-overwrite window, so behavior is identical
//!   modulo wire latency).

#![deny(clippy::unwrap_used)]

use std::{net::SocketAddr, sync::Arc};

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    InfinoError,
    config::{
        CompactionSettings, Config, MemorySettings, StorageBackend, StorageColdFetchMode,
        StorageSettings, SupertableSettings,
    },
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{
        Supertable,
        query::VectorSearchOptions,
        storage::{S3StorageProvider, StorageProvider},
    },
    test_helpers::{
        build_title_batch, default_disk_cache, default_supertable_options,
        lazy_foreground_disk_cache,
    },
};

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
/// BM25 top-k for the smoke FTS query.
const BM25_TOP_K: usize = 10;
/// Connection memory budget for the over-budget e2e: 1 byte. The 90%
/// gate floors to 0, so the first cold cluster-block fetch (which is
/// always more than 0 bytes) is refused. Any positive value would do;
/// 1 makes "the smallest possible allowance still denies" explicit.
const TINY_BUDGET_BYTES: u64 = 1;
/// Row count for the over-budget e2e fixture. Large enough that the IVF
/// cluster blocks are a genuine cold object-store fetch rather than being
/// swallowed by the lazy reader's open-range prefetch (which, on the tiny
/// 8-row smoke fixture, would leave the whole vector subsection resident
/// → warm → not gated). A few thousand rows puts real bytes in each
/// cluster block.
const BUDGET_N_ROWS: usize = 4096;
/// Expected peak reservation for the measured control's cold vector search.
/// The cold cluster-block fetch over the `BUDGET_N_ROWS` fixture (dim 16,
/// `n_cent` 4, Sq8 rerank, `nprobe` 4) is a deterministic 126,464 B. Assert
/// a band around it: tight enough to prove it's the real cluster fetch (not
/// a stray small read), loose enough to survive minor codec / layout drift.
const CONTROL_PEAK_LOW_BYTES: usize = 100_000;
const CONTROL_PEAK_HIGH_BYTES: usize = 160_000;
/// A bounded (enforcing) budget set generously above one cold fetch. Proves an
/// enforcing budget admits an under-budget query rather than refusing on
/// principle: the 90% gate is 900 KB, far above the ~126 KB fetch. Distinct
/// from the measured control, whose `None` limit can never deny by construction.
const AMPLE_BUDGET_BYTES: u64 = 1_000_000;
/// The 90% gate `AMPLE_BUDGET_BYTES` resolves to (`limit()` returns the gate,
/// not the raw config value). Asserted so the test proves the budget is truly
/// bounded, not silently measured.
const AMPLE_BUDGET_GATE_BYTES: usize = 900_000;
/// Connection budget for the shared-budget (multi-superfile) e2e. Sized to
/// admit one superfile's ~126 KB cold cluster-block fetch but not two at once:
/// the 90% gate is ~180 KB, so a single fetch fits and the two concurrent
/// fetches of the fan-out (~253 KB together) cross it. This is what proves the
/// budget is shared across superfiles rather than per-superfile.
const SHARED_BUDGET_BYTES: u64 = 200_000;
use s3s::{auth::SimpleAuth, service::S3ServiceBuilder};
use s3s_fs::FileSystem;
use tempfile::TempDir;
use tokio::net::TcpListener;

const TEST_BUCKET: &str = "infino-s3-smoke";
const TEST_REGION: &str = "us-east-1";
const TEST_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const TEST_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

/// Spawn s3s-fs on a random port. Returns the bound
/// address + the tempdir guard (must stay alive for the
/// test's lifetime — drop unlinks the bucket data).
async fn spawn_s3s_fs() -> (SocketAddr, TempDir) {
    let fs_root = TempDir::new().expect("s3s-fs root tempdir");
    // s3s-fs treats top-level dirs as buckets. Pre-create
    // the bucket dir so put_atomic on a key inside it
    // doesn't 404 the bucket itself.
    std::fs::create_dir_all(fs_root.path().join(TEST_BUCKET)).expect("create bucket dir");

    let fs_backend = FileSystem::new(fs_root.path()).expect("s3s-fs FileSystem");
    // Configure auth so s3s accepts the SigV4-signed
    // requests object_store sends. Without `set_auth`, s3s
    // responds 501 "no authentication provider" to any
    // signed request.
    let service = {
        let mut b = S3ServiceBuilder::new(fs_backend);
        b.set_auth(SimpleAuth::from_single(TEST_ACCESS_KEY, TEST_SECRET_KEY));
        b.build()
    };
    // S3Service derives Clone (internally Arc<Inner>); clones
    // share the underlying service handle.

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");

    tokio::spawn(async move {
        use hyper_util::{
            rt::{TokioExecutor, TokioIo},
            server::conn::auto::Builder as ConnBuilder,
        };
        let http = ConnBuilder::new(TokioExecutor::new());
        loop {
            let (stream, _peer) = match listener.accept().await {
                Ok(t) => t,
                Err(_) => break,
            };
            let service = service.clone();
            let http = http.clone();
            tokio::spawn(async move {
                let _ = http.serve_connection(TokioIo::new(stream), service).await;
            });
        }
    });

    (addr, fs_root)
}

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
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8ResidualEpsilon,
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
        memory: MemorySettings::default(),
    }
}

/// A `Config` that carries only a connection memory budget; storage
/// backend is `None` so `apply_config` leaves the storage / disk cache
/// unattached (the caller wires those explicitly afterward). Drives the
/// bounded budget onto options via the same `apply_config` path a
/// `config.yaml`-built connection takes.
fn budget_only_config(connection_budget_bytes: u64) -> Config {
    Config {
        supertable: SupertableSettings::default(),
        storage: StorageSettings {
            backend: StorageBackend::None,
            ..StorageSettings::default()
        },
        compaction: CompactionSettings::default(),
        memory: MemorySettings {
            connection_budget_bytes,
        },
    }
}

/// An `S3StorageProvider` pointed at the in-process s3s-fs `endpoint`, boxed as
/// a trait object. Every budget-e2e handle (producer + consumers) reaches the
/// same bucket through one of these.
fn budget_s3_provider(endpoint: &str) -> Arc<dyn StorageProvider> {
    Arc::new(
        S3StorageProvider::new_with_endpoint(
            endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("s3 provider for budget e2e"),
    )
}

/// Open a fresh consumer against `storage` with a lazy-foreground disk cache
/// (so vector reads stay cold / non-resident) and `connection_budget_bytes` as
/// the connection budget (0 = measured). Returns the handle plus the cache's
/// `TempDir` guard: it must outlive the query, since dropping it unlinks the
/// cache files the reader is still reading through.
fn open_budget_consumer(
    dim: usize,
    storage: &Arc<dyn StorageProvider>,
    connection_budget_bytes: u64,
) -> (Supertable, TempDir) {
    let cache_dir = TempDir::new().expect("budget consumer cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(storage), cache_dir.path());
    let consumer = Supertable::open(
        real_s3_options(dim)
            .apply_config(&budget_only_config(connection_budget_bytes))
            .expect("apply budget config to consumer options")
            .with_storage(Arc::clone(storage))
            .with_disk_cache(cache),
    )
    .expect("Supertable::open via S3 (budget consumer)");
    (consumer, cache_dir)
}

/// A larger vector+FTS batch for the over-budget e2e: `BUDGET_N_ROWS`
/// rows so the IVF cluster blocks carry real bytes (see
/// [`BUDGET_N_ROWS`]). Each row's embedding is a one-hot at `row % dim`
/// (enough spread for `n_cent` clusters); the title carries the row
/// index so FTS has content too.
fn budget_vector_batch(dim: usize, n_rows: usize) -> RecordBatch {
    let titles = LargeStringArray::from(
        (0..n_rows)
            .map(|i| format!("budget vector row {i}"))
            .collect::<Vec<_>>(),
    );
    let mut flat = Vec::with_capacity(n_rows * dim);
    for row in 0..n_rows {
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
async fn supertable_smoke_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_smoke_via_s3_wire_protocol: skipped (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    eprintln!("[s3-smoke] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Quick provider-level smoke before invoking the full
    // writer path — isolates "the S3 provider works at all"
    // from "the writer + cache stack works on top".
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for probe"),
        );
        let probe_bytes = bytes::Bytes::from_static(b"hello-smoke");
        storage
            .put_atomic("probe/hello.txt", probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
        assert_eq!(got, probe_bytes, "probe round-trip mismatch");
        eprintln!("[s3-smoke] probe round-trip OK (PUT + GET via S3 wire)");
    }

    // Producer: writes through the S3 wire protocol.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for producer"),
        );
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("producer writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("producer commit via S3");
        assert_eq!(producer.manifest_id(), 1);
        eprintln!(
            "[s3-smoke] producer commit OK; manifest_id={}",
            producer.manifest_id()
        );
    }

    // Consumer: opens via the same S3 endpoint + a disk
    // cache. Reads should route through the cache → S3
    // get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        S3StorageProvider::new_with_endpoint(
            &endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("s3 provider for consumer"),
    );
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = default_disk_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via S3");

    assert_eq!(consumer.manifest_id(), 1, "recovered manifest_id mismatch");
    assert_eq!(
        consumer.reader().n_docs_total(),
        2,
        "recovered n_docs_total mismatch"
    );
    eprintln!(
        "[s3-smoke] consumer open OK; manifest_id={} n_superfiles={} n_docs_total={}",
        consumer.manifest_id(),
        consumer.reader().n_superfiles(),
        consumer.reader().n_docs_total()
    );

    // SQL query through cache. First query cold-fetches via
    // S3; n_cold_fetches grows.
    let pre = cache.stats();
    assert_eq!(pre.n_cold_fetches, 0);
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query_sql via S3");
    assert_eq!(batches.len(), 1);
    let post = cache.stats();
    assert!(
        post.n_cold_fetches >= 1,
        "first query must cold-fetch through S3; got n_cold_fetches={}",
        post.n_cold_fetches
    );
    eprintln!(
        "[s3-smoke] cold-fetch via S3 OK; n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );

    eprintln!("[s3-smoke] smoke done");
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
    result.expect("real S3 integration failed");
}

/// TVF lane over the S3 wire protocol: exercises
/// `bm25_search`, `vector_search`, and `hybrid_search`
/// end-to-end through `query_sql` (DataFusion plan -> custom
/// `TableProvider` -> custom exec -> kernel -> resolve to
/// `_id`) against an S3-backed supertable. The existing
/// `supertable_smoke_via_s3_wire_protocol` covers
/// `SELECT COUNT(*)` (provider scan path); this one covers the
/// search TVFs, which is where the retrieval engine actually
/// earns its keep on object storage.
///
/// Asserts `cache.stats().n_cold_fetches` grew across the
/// three queries — proves the TVF reads went through the
/// s3s-fs server (HTTP get_range), not a local short-circuit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_tvfs_through_query_sql_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_tvfs_through_query_sql_via_s3_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = EMB_DIM;
    eprintln!("[s3-smoke-tvf] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Producer: writes a title (FTS) + emb (vector) batch
    // through the S3 wire protocol.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for tvf producer"),
        );
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create tvf producer");
        let mut w = producer.writer().expect("tvf producer writer");
        w.append(&real_s3_batch(dim))
            .expect("append unified vector+FTS batch");
        w.commit().expect("tvf producer commit via S3");
        assert_eq!(producer.manifest_id(), 1);
    }

    // Consumer: opens via the same S3 endpoint + a disk
    // cache. TVF reads cold-fetch through HTTP get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        S3StorageProvider::new_with_endpoint(
            &endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("s3 provider for tvf consumer"),
    );
    let cache_dir = TempDir::new().expect("tvf cache tempdir");
    let cache = default_disk_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        real_s3_options(dim)
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via S3 (tvf consumer)");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), EXPECTED_N_DOCS);

    let pre = cache.stats();

    // One-hot query vector at dim 0. `real_s3_batch` row 0
    // has emb[0]=1.0, so doc 0 is the closest vector match.
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    fn count_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    // 1. bm25_search through query_sql. The corpus has "alpha"
    //    in exactly two titles ("alpha vector one", "alpha
    //    vector two"), so the TVF must return >= 2 rows.
    let bm25 = consumer
        .reader()
        .query_sql(&format!(
            "SELECT _id FROM bm25_search('title', 'alpha', {BM25_TOP_K})"
        ))
        .expect("bm25_search via query_sql over S3");
    assert!(
        count_rows(&bm25) >= 2,
        "bm25_search('alpha') should return >=2 docs over S3; got {}",
        count_rows(&bm25)
    );

    // 2. vector_search through query_sql. k=3.
    let vec_sql = format!("SELECT _id FROM vector_search('emb', '{q_csv}', 3)");
    let vector = consumer
        .reader()
        .query_sql(&vec_sql)
        .expect("vector_search via query_sql over S3");
    assert!(
        count_rows(&vector) >= 1,
        "vector_search returned no rows over S3"
    );

    // 3. hybrid_search through query_sql. RRF fusion over the
    //    same two retrievers; k=5.
    let hybrid_sql =
        format!("SELECT _id FROM hybrid_search('title', 'alpha', 'emb', '{q_csv}', 5)");
    let hybrid = consumer
        .reader()
        .query_sql(&hybrid_sql)
        .expect("hybrid_search via query_sql over S3");
    let hyb_rows = count_rows(&hybrid);
    assert!(
        hyb_rows > 0 && hyb_rows <= 5,
        "hybrid_search rows in (0, 5]; got {hyb_rows}"
    );

    // 4. Cold-fetch counter grew -> confirms TVF reads went
    //    through the S3 wire path, not a local short-circuit.
    let post = cache.stats();
    assert!(
        post.n_cold_fetches > pre.n_cold_fetches,
        "TVF queries must cold-fetch through S3; pre={} post={}",
        pre.n_cold_fetches,
        post.n_cold_fetches
    );

    eprintln!(
        "[s3-smoke-tvf] bm25 / vector / hybrid via query_sql over S3 OK; \
         n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );
}

/// A cold vector search over the S3 wire protocol, under a bounded
/// per-connection memory budget, is refused with `InfinoError::OverBudget`.
///
/// The consumer opens with `TINY_BUDGET_BYTES` (bounded) and a
/// lazy-foreground disk cache, so the foreground vector query reads through
/// a `StorageRangeSource`:
///  - the superfile's byte source is never resident, so the cold
///    cluster-block fetch is a real object-store GET.
///  - that GET reserves against the budget first; the reservation exceeds
///    the (floored-to-0) gate and is refused before the GET is issued.
///  - the refusal propagates the full chain to the public error:
///    cold-fetch reserve (deny) -> VectorError::OverBudget ->
///    ReadError::Vector -> QueryError::OverBudget -> InfinoError::OverBudget.
///
/// A measured (unbounded) control then runs the identical cold query to
/// completion, so deny-on-cold is the only behavioral change the budget
/// introduces. Warm (resident) queries are never gated.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_vector_search_over_budget_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_cold_vector_search_over_budget_via_s3_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = EMB_DIM;
    eprintln!("[s3-budget] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Producer: commit a vector batch through the S3 wire protocol.
    let storage = budget_s3_provider(&endpoint);
    {
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via S3");
        assert_eq!(producer.manifest_id(), 1);
    }

    // Consumer: fresh handle + lazy-foreground disk cache (reads stay
    // non-resident) under a bounded connection budget. `_cache_guard` keeps
    // the cache files alive for the query.
    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), BUDGET_N_ROWS as u64);

    // One-hot query at dim 0. Under a measured budget row 0 is the
    // closest match; here the cold cluster-block fetch must be refused
    // before the vector shortlist runs.
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let result = consumer.vector_search(
        "emb",
        &q,
        VECTOR_SEARCH_K,
        VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
        None,
        None,
    );

    match result {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[s3-budget] cold vector search refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(hits) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold vector search returned {} batch(es)",
            hits.len()
        ),
    }

    // Budget accounting on the refused path: the gate fired (>=1 denial) and,
    // because a refused reservation commits nothing, peak usage stays 0.
    let bounded_budget = consumer.options().connection_budget();
    eprintln!(
        "[s3-budget] bounded budget: denials={} peak={} B",
        bounded_budget.denials(),
        bounded_budget.peak()
    );
    assert!(
        bounded_budget.denials() >= 1,
        "bounded budget must record >=1 denial; got {}",
        bounded_budget.denials()
    );
    assert_eq!(
        bounded_budget.peak(),
        0,
        "a refused cold fetch commits nothing, so peak must stay 0"
    );

    // Control: the identical cold query over the identical fixture runs to
    // completion under a measured (unbounded) budget. Proves the deny above
    // is caused by the budget, not by a broken fixture or a cold-path error.
    // Fresh consumer + fresh cache so the read is cold again.
    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_hits = control
        .vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("measured cold vector search should run to completion");
    let control_rows: usize = control_hits.iter().map(|b| b.num_rows()).sum();
    assert!(
        control_rows >= 1,
        "measured cold vector search returned no rows over S3"
    );
    // The measured budget never refuses, so the same cold cluster-block fetch
    // that the bounded budget rejected is here reserved + released: peak > 0
    // proves the reservation actually ran on the query path (never denied).
    let control_budget = control.options().connection_budget();
    eprintln!(
        "[s3-budget] measured control: rows={control_rows} denials={} peak={} B",
        control_budget.denials(),
        control_budget.peak()
    );
    assert_eq!(
        control_budget.denials(),
        0,
        "measured budget must never deny"
    );
    let control_peak = control_budget.peak();
    assert!(
        (CONTROL_PEAK_LOW_BYTES..=CONTROL_PEAK_HIGH_BYTES).contains(&control_peak),
        "measured cold vector search peak {control_peak} B outside expected \
         [{CONTROL_PEAK_LOW_BYTES}, {CONTROL_PEAK_HIGH_BYTES}] band; \
         a peak near 0 means the budget was never exercised on the query path"
    );

    // Bounded but ample: an enforcing budget well above one fetch must admit
    // the query. A bounded budget refuses only when a reservation would cross
    // the gate, not on principle, so the same cold fetch that a 1-byte budget
    // rejected runs here, reserves, and never denies.
    let (ample, _ample_guard) = open_budget_consumer(dim, &storage, AMPLE_BUDGET_BYTES);
    let ample_budget = ample.options().connection_budget();
    // The limit is set (not `None`), so this is genuinely bounded, not measured.
    assert_eq!(
        ample_budget.limit(),
        Some(AMPLE_BUDGET_GATE_BYTES),
        "ample budget must be bounded (an enforced gate), not measured"
    );
    let ample_hits = ample
        .vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("under-budget cold vector search should run under a bounded budget");
    let ample_rows: usize = ample_hits.iter().map(|b| b.num_rows()).sum();
    let ample_peak = ample_budget.peak();
    eprintln!(
        "[s3-budget] bounded-ample: rows={ample_rows} denials={} peak={ample_peak} B",
        ample_budget.denials()
    );
    assert!(
        ample_rows >= 1,
        "bounded-ample cold vector search returned no rows"
    );
    assert_eq!(
        ample_budget.denials(),
        0,
        "an under-budget query must not be denied by a bounded budget"
    );
    assert!(
        (CONTROL_PEAK_LOW_BYTES..=CONTROL_PEAK_HIGH_BYTES).contains(&ample_peak),
        "bounded-ample peak {ample_peak} B outside expected \
         [{CONTROL_PEAK_LOW_BYTES}, {CONTROL_PEAK_HIGH_BYTES}] band"
    );
}

/// The same cold-fetch budget refusal, but the vector search is embedded in a
/// SQL query (the `vector_search` table function) instead of the direct API.
///
/// The kNN runs in a custom execution node, so its `QueryError::OverBudget`
/// has to cross the DataFusion error boundary and back. The node maps it to
/// `DataFusionError::ResourcesExhausted` (the same channel the SQL memory pool
/// uses for a spill refusal), so the SQL error classifier routes it to the
/// public `InfinoError::OverBudget` rather than flattening it to a generic
/// query error. A measured control runs the identical SQL to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_vector_search_over_budget_via_sql_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_cold_vector_search_over_budget_via_sql_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = EMB_DIM;

    // Producer: commit a vector batch through the S3 wire protocol.
    let storage = budget_s3_provider(&endpoint);
    {
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via S3");
    }

    // One-hot query at dim 0 as the SQL table-function's CSV argument.
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT _id FROM vector_search('emb', '{q_csv}', {VECTOR_SEARCH_K})");

    // Cold consumer under a bounded 1-byte budget: the kNN's cold cluster-block
    // fetch is refused, and the refusal must survive the SQL boundary as
    // `InfinoError::OverBudget`. `map_err(InfinoError::from)` is the same
    // conversion the curated public surface applies.
    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    let result = consumer.reader().query_sql(&sql).map_err(InfinoError::from);
    match result {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[s3-budget-sql] cold vector search in SQL refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(batches) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold vector search in SQL returned {} batch(es)",
            batches.len()
        ),
    }
    let bounded_budget = consumer.options().connection_budget();
    assert!(
        bounded_budget.denials() >= 1,
        "bounded budget must record >=1 denial; got {}",
        bounded_budget.denials()
    );

    // Control: the identical SQL over a measured (unbounded) budget runs to
    // completion, so the deny above is the budget's doing, not a broken plan.
    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_rows: usize = control
        .reader()
        .query_sql(&sql)
        .expect("measured cold vector search in SQL should run to completion")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert!(
        control_rows >= 1,
        "measured cold vector search in SQL returned no rows over S3"
    );
    assert_eq!(
        control.options().connection_budget().denials(),
        0,
        "measured budget must never deny"
    );
}

/// Same cold-fetch budget refusal, but through the `hybrid_search` table
/// function (BM25 + vector fused with RRF). Hybrid runs the vector kNN in its
/// own execution node, distinct from `vector_search`, so it has its own error
/// boundary. This pins that the vector arm's `OverBudget` survives that node
/// too and isn't flattened to a generic query error. A measured control runs
/// the identical SQL to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_hybrid_search_over_budget_via_sql_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_cold_hybrid_search_over_budget_via_sql_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = EMB_DIM;

    let storage = budget_s3_provider(&endpoint);
    {
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via S3");
    }

    // 'budget' is in every title (see budget_vector_batch), so BM25 matches;
    // the vector arm is what crosses the budget on its cold cluster fetch.
    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT _id FROM hybrid_search('title', 'budget', 'emb', '{q_csv}', {VECTOR_SEARCH_K})"
    );

    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    match consumer.reader().query_sql(&sql).map_err(InfinoError::from) {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[s3-budget-sql] cold hybrid search refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(batches) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold hybrid search returned {} batch(es)",
            batches.len()
        ),
    }
    assert!(
        consumer.options().connection_budget().denials() >= 1,
        "bounded budget must record >=1 denial"
    );

    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_rows: usize = control
        .reader()
        .query_sql(&sql)
        .expect("measured cold hybrid search should run to completion")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert!(
        control_rows >= 1,
        "measured cold hybrid search returned no rows over S3"
    );
    assert_eq!(
        control.options().connection_budget().denials(),
        0,
        "measured budget must never deny"
    );
}

/// The per-connection budget is shared across the multi-superfile fan-out, so
/// several superfiles' cold fetches accumulate against one ceiling. A
/// per-superfile budget would never add up, and this is what pins that.
///
/// Two commits produce two superfiles, each with its own ~126 KB cold
/// cluster-block fetch. The unfiltered fan-out probes them concurrently, so
/// both reservations are live against the same budget at once:
///  - measured (unbounded): both fetch, the query runs, and peak climbs past a
///    single superfile's fetch (the two live reservations sum on one budget).
///  - bounded to `SHARED_BUDGET_BYTES` (fits one fetch, not two): the second
///    concurrent reservation crosses the ceiling and the query is refused as
///    `InfinoError::OverBudget`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_vector_budget_is_shared_across_superfiles_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_vector_budget_is_shared_across_superfiles_via_s3_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = EMB_DIM;
    eprintln!("[s3-shared] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Producer: two commits => two superfiles, each `BUDGET_N_ROWS` rows.
    let storage = budget_s3_provider(&endpoint);
    {
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create shared-budget producer");
        for commit in 0..2 {
            let mut w = producer.writer().expect("shared-budget producer writer");
            w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
                .expect("append large vector+FTS batch");
            w.commit().expect("shared-budget producer commit via S3");
            assert_eq!(producer.manifest_id(), commit + 1);
        }
    }

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let search = |table: &Supertable| {
        table.vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
    };

    // Measured: both superfiles fetch, the query runs, and the peak exceeds a
    // single superfile's fetch => the two reservations summed on one budget.
    let (measured, _measured_guard) = open_budget_consumer(dim, &storage, 0);
    assert_eq!(measured.reader().n_superfiles(), 2);
    assert_eq!(measured.reader().n_docs_total(), (BUDGET_N_ROWS as u64) * 2);
    let measured_hits = search(&measured).expect("measured search over two superfiles runs");
    let measured_rows: usize = measured_hits.iter().map(|b| b.num_rows()).sum();
    let measured_peak = measured.options().connection_budget().peak();
    eprintln!("[s3-shared] measured: rows={measured_rows} peak={measured_peak} B");
    assert!(
        measured_rows >= 1,
        "measured two-superfile search returned no rows"
    );
    assert!(
        measured_peak > CONTROL_PEAK_HIGH_BYTES,
        "peak {measured_peak} B should exceed one superfile's fetch \
         ({CONTROL_PEAK_HIGH_BYTES} B): the two fetches must sum on one budget"
    );

    // Bounded to fit one fetch but not two: the second concurrent reservation
    // crosses the ceiling, so the shared budget refuses the query.
    let (bounded, _bounded_guard) = open_budget_consumer(dim, &storage, SHARED_BUDGET_BYTES);
    let result = search(&bounded);
    let bounded_budget = bounded.options().connection_budget();
    eprintln!(
        "[s3-shared] bounded: denials={} peak={} B result={}",
        bounded_budget.denials(),
        bounded_budget.peak(),
        if result.is_ok() { "ok" } else { "over-budget" }
    );
    match result {
        Err(InfinoError::OverBudget(_)) => {}
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(hits) => panic!(
            "two concurrent {CONTROL_PEAK_HIGH_BYTES}-B fetches must cross a \
             {SHARED_BUDGET_BYTES}-B budget; got {} batch(es)",
            hits.len()
        ),
    }
    assert!(
        bounded_budget.denials() >= 1,
        "the shared budget must record the crossing"
    );
}
