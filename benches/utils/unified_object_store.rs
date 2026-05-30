//! Plan 013 M5 — unified vector + FTS cold-open / cold-first-search
//! / warm-search against an in-process S3 server (`s3s-fs`).
//!
//! Spawns `s3s-fs` on a random port, points an
//! `S3StorageProvider` at it, uploads a real **unified**
//! superfile (one Parquet file carrying both a vector
//! subsection and an FTS subsection — the "consolidated
//! vector / fts data layer" shape), and runs the cold-open /
//! cold-first-search / warm-search rows for *both* structures
//! through the *same* `DiskCacheStore` +
//! `ColdFetchMode::LazyForegroundWithBackgroundFill` path:
//!
//! 1. **Cold open via S3** — `cache.reader(uri)` against an
//!    empty cache; pays the Plan 013 cold-open budget (Parquet
//!    footer + per-subsection open-time-region GETs). One open
//!    serves both the vector and FTS readers.
//! 2. **Cold first vector search after S3 open** — cold open +
//!    `vec.search` at the default `(nprobe, rerank_mult)`;
//!    pays the M3 cold-search budget (~nprobe + 1 cluster
//!    GETs).
//! 3. **Cold first BM25 search after S3 open** — cold open +
//!    `bm25_search`; pays the FTS lazy open-time fetch
//!    (header + doc-lengths) plus per-term dict/postings range
//!    GETs (`FtsReader::open_lazy` mirroring the vector path).
//! 4. **Warm subsequent search after S3 open** — after the
//!    background promotion completes, the cache returns the
//!    mmap-backed reader and both vector + BM25 searches
//!    resolve entirely from mmap (zero S3 GETs).
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench object-store                      # default scale (100k)
//! INFINO_BENCH_FULL=1 cargo bench --bench object-store  # 1M scale (README row)
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench object-store  # also rewrite README
//! ```
//!
//! Default scale is 100k × 384 (fast iteration, ~150 MiB
//! superfile). `INFINO_BENCH_FULL=1` bumps to 1M × 384
//! (~1.5 GiB), which is the headline number in
//! `benches/vector/README.md`.
//!
//! Throughput rows always print to stderr via the shared
//! `emit_*_markdown()` pattern; `INFINO_BENCH_UPDATE_README=1`
//! additionally rewrites the matching section in
//! `benches/vector/README.md`.
//!
//! ## Why s3s-fs (plus adjusted reporting, not LocalFs-only)
//!
//! - `LocalFsStorageProvider`'s `get_range` is a `pread64`;
//!   the request never crosses an HTTP boundary, so the
//!   measurement misses every effect the production code
//!   pays (HTTP round-trip, range parsing, byte-range
//!   header encoding, connection reuse).
//! - Real AWS S3 has region-dependent + time-dependent p50
//!   tails that distort a regression bench.
//! - `s3s-fs` gives us the full S3 wire path (path-style URL
//!   + SigV4 + HTTP/1.1 range headers), so it is useful for
//!   validating request shape: GET count, byte ranges, and
//!   overlap. Its loopback latency is not treated as S3 latency;
//!   the diagnostic prints an adjusted model line that replaces
//!   the observed s3s-fs blocking span with a configurable S3
//!   TTFB + throughput model.

#![allow(clippy::too_many_arguments)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use criterion::{Criterion, criterion_group};
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::SuperfileUri;
use infino::supertable::reader_cache::{
    ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy,
};
use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use infino::test_helpers::default_tokenizer;
use s3s::auth::SimpleAuth;
use s3s::service::S3ServiceBuilder;
use s3s_fs::FileSystem;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

// ─── Constants ───────────────────────────────────────────────────────

const TEST_BUCKET: &str = "infino-013-bench";
const TEST_REGION: &str = "us-east-1";
const TEST_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const TEST_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

/// Doc count. 100k × 384 is ~150 MiB on disk — small enough
/// to iterate quickly on s3s-fs, large enough that the
/// per-cluster range fetches dominate the search path.
/// `INFINO_BENCH_FULL=1` bumps to 1M for the headline
/// README row.
fn n_docs() -> usize {
    if std::env::var("INFINO_BENCH_FULL").is_ok() {
        1_000_000
    } else {
        100_000
    }
}

/// Default `(nprobe, rerank_mult)` for the search rows.
/// Matches the production default in `VectorSearchOptions`.
const DEFAULT_NPROBE: usize = 8;
const DEFAULT_RERANK_MULT: usize = 20;
const TOP_K: usize = 10;

/// Primary-key column. `SuperfileBuilder` requires the id column
/// to be `Decimal128(38, 0)` (the supertable's snowflake id type).
const ID_COLUMN: &str = "doc_id";
/// Vector column logical name (lives only in the embedded vector
/// blob, not the Parquet schema).
const VEC_COLUMN: &str = "v";
/// FTS column registered on the unified fixture. Single `title`
/// column — same shape `benches/utils/fts_superfile.rs` builds.
/// It stays in the Parquet body (SQL-visible) *and* is indexed
/// into the FTS blob, which is the whole point of the unified
/// layout.
const FTS_COLUMN: &str = "title";
/// Zipfian-common term (`MmapTextCorpus` plants `term00001` as
/// the highest-frequency term), so the cold BM25 row exercises a
/// real multi-block postings fetch rather than a df=1 sliver.
const FTS_QUERY_TERM: &str = "term00001";

// ─── Fixtures (built once per `cargo bench` invocation) ──────────────

static SUPERFILE_BYTES: OnceLock<Bytes> = OnceLock::new();
static QUERY_VECTOR: OnceLock<Vec<f32>> = OnceLock::new();

fn superfile_bytes() -> &'static Bytes {
    SUPERFILE_BYTES.get_or_init(build_superfile_bytes)
}

fn query_vector() -> &'static [f32] {
    QUERY_VECTOR
        .get_or_init(|| {
            let n = n_docs();
            let v = crate::corpus::MmapVectorCorpus::generate(n, crate::corpus::n_cent(n), 1, true);
            // Take vector at index 0 as the query — known to
            // exist in the planted-cluster corpus + a real-
            // shape query (not orthogonal to every cluster).
            v.as_slice()[..crate::corpus::DIM].to_vec()
        })
        .as_slice()
}

/// Build a real **unified** superfile (one vector column + one FTS
/// column over the same docs) by driving the production
/// [`SuperfileBuilder`] — the exact path the supertable writer takes
/// at commit. The bench owns **no** format logic: it only feeds Arrow
/// batches + vector slices and lets the builder produce the FTS index,
/// the IVF/RaBitQ vector blob, the Parquet body, the blob splice, and
/// the `inf.*` KV metadata. Cached in `SUPERFILE_BYTES` for the
/// bench's lifetime so every row shares one fixture.
fn build_superfile_bytes() -> Bytes {
    let n = n_docs();
    let n_cent = crate::corpus::n_cent(n);
    let dim = crate::corpus::DIM;

    let vectors_mmap = crate::corpus::MmapVectorCorpus::generate(n, n_cent, 1, true);
    let vectors = vectors_mmap.as_slice();
    let text = crate::corpus::MmapTextCorpus::generate(n, 1);

    // Schema = id (Decimal128, as the supertable injects) + the FTS
    // text column. The vector column is a logical name only; its f32
    // buffer is passed alongside each batch, not as a schema field.
    let schema = Arc::new(Schema::new(vec![
        Field::new(ID_COLUMN, DataType::Decimal128(38, 0), false),
        Field::new(FTS_COLUMN, DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![FtsConfig {
            column: FTS_COLUMN.into(),
        }],
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8,
        }],
        Some(default_tokenizer()),
    );
    let mut builder = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    eprintln!(
        "[object_store_bench] building {n}-doc unified superfile \
         (vector n_cent={n_cent} + FTS `{FTS_COLUMN}`) via SuperfileBuilder"
    );
    let t0 = Instant::now();

    // Feed the corpus in row-group-sized chunks so neither a 1M-row
    // Arrow batch nor a whole-corpus `Vec<String>` is ever resident —
    // the mmap corpora stay the only large allocation.
    const CHUNK: usize = 65_536;
    let mut start = 0;
    while start < n {
        let len = CHUNK.min(n - start);
        let ids: Decimal128Array = (start as u64..(start + len) as u64)
            .map(|i| Some(i as i128))
            .collect::<Decimal128Array>()
            .with_precision_and_scale(38, 0)
            .expect("decimal128 with_precision_and_scale");
        let titles =
            LargeStringArray::from((start..start + len).map(|i| text.doc(i)).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");
        builder
            .add_batch(&batch, &[&vectors[start * dim..(start + len) * dim]])
            .expect("add_batch");
        start += len;
    }

    let bytes = builder.finish().expect("finish SuperfileBuilder");
    eprintln!(
        "[object_store_bench] unified superfile built: {} MiB in {:.1}s",
        bytes.len() / (1024 * 1024),
        t0.elapsed().as_secs_f32(),
    );
    Bytes::from(bytes)
}

// ─── S3 latency model (Plan 013 #1: adjusted diagnostic) ────────────
//
// `s3s-fs` over loopback faithfully reproduces the S3 *request
// count* and *byte volume* (the things Plan 013's GET-minimization
// optimizes), but not S3's *wall-clock* — its per-request RTT in
// this sandbox is dominated by a fixed ~650 ms artifact that has
// nothing to do with real S3. To get a meaningful cold-open /
// cold-search wall-clock signal while iterating (before the gated
// real-S3 suite in PR9 gives the ground truth), the diagnostic
// reports a synthetic AWS-S3-in-region timing model on top of the
// real request shape:
//
//   wall(req) = TTFB + bytes / throughput
//
// TTFB models the round-trip + first-byte latency S3 charges per
// request regardless of size; the throughput term models single-
// stream transfer bandwidth. The diagnostic does not sleep in
// the measured path. It records actual s3s-fs request intervals,
// subtracts their observed blocking span from wall-clock, and adds
// back this model grouped by the same observed parallel batches.
//
// These knobs affect only the diagnostic's adjusted/modelled line;
// they never alter the code under measurement.
//
//   INFINO_S3_MODEL_TTFB_MS=<f64>  per-request first-byte latency
//                                  (default 100 ms)
//   INFINO_S3_MODEL_MBPS=<f64>     single-stream throughput in MB/s
//                                  (default 100 MB/s — single-object
//                                  cold-read floor; aggregate multi-
//                                  key throughput is far higher but
//                                  irrelevant to one cold object)
#[derive(Debug, Clone, Copy)]
struct S3LatencyModel {
    ttfb: Duration,
    bytes_per_sec: f64,
}

impl S3LatencyModel {
    /// Read the model used for adjusted diagnostic reporting.
    /// This never changes the measured code path.
    fn from_env_or_default() -> Self {
        let ttfb_ms = std::env::var("INFINO_S3_MODEL_TTFB_MS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);
        let mbps = std::env::var("INFINO_S3_MODEL_MBPS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);
        Self {
            ttfb: Duration::from_secs_f64(ttfb_ms / 1000.0),
            bytes_per_sec: mbps * 1_000_000.0,
        }
    }

    fn delay_for(&self, bytes: u64) -> Duration {
        self.ttfb + Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec)
    }
}

// ─── s3s-fs harness ──────────────────────────────────────────────────

/// Spawn s3s-fs on a random loopback port. Returns the bound
/// addr + the tempdir that owns the FS root (kept alive by
/// the caller).
async fn spawn_s3s_fs() -> (SocketAddr, TempDir) {
    let fs_root = TempDir::new().expect("s3s-fs root tempdir");
    std::fs::create_dir_all(fs_root.path().join(TEST_BUCKET)).expect("create bucket dir");

    let fs_backend = FileSystem::new(fs_root.path()).expect("s3s-fs FileSystem");
    let service = {
        let mut b = S3ServiceBuilder::new(fs_backend);
        b.set_auth(SimpleAuth::from_single(TEST_ACCESS_KEY, TEST_SECRET_KEY));
        b.build()
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");

    tokio::spawn(async move {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
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

/// One-time s3s-fs setup: spawn server, upload superfile,
/// return the storage handle + URI to query. The tempdir
/// stays alive in the returned tuple — drop it after the
/// bench to GC the bucket data.
async fn setup_s3_fixture(
    superfile: &Bytes,
) -> (SocketAddr, TempDir, Arc<dyn StorageProvider>, SuperfileUri) {
    let (addr, fs_root) = spawn_s3s_fs().await;
    let endpoint = format!("http://{addr}");
    let storage: Arc<dyn StorageProvider> = Arc::new(
        S3StorageProvider::new_with_endpoint(
            &endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("S3StorageProvider"),
    );
    let uri = SuperfileUri::new_v4();
    let path = format!("data/seg-{}.sf", uri.0);
    // Upload against the raw provider — fixture-setup latency is
    // not part of the measured cold path.
    storage
        .put_atomic(&path, superfile.clone())
        .await
        .expect("upload superfile to s3s-fs");
    eprintln!(
        "[object_store_bench] s3s-fs spawned on {endpoint}, superfile uploaded to {path}"
    );
    (addr, fs_root, storage, uri)
}

/// Fresh disk-cache in `LazyForegroundWithBackgroundFill` mode.
/// Returns the cache + its temp root (drop after to GC).
fn fresh_cache(storage: Arc<dyn StorageProvider>) -> (TempDir, Arc<DiskCacheStore>) {
    let dir = TempDir::new().expect("cache tempdir");
    let cfg = DiskCacheConfig {
        cache_root: dir.path().to_path_buf(),
        disk_budget_bytes: 4 * (1u64 << 30),
        cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
        cold_fetch_streams: 8,
        cold_fetch_chunk_bytes: 4 * (1u64 << 20),
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: false,
    };
    let store = DiskCacheStore::new_unpinned(storage, cfg).expect("DiskCacheStore");
    (dir, store)
}

/// Poll the cache until the URI is promoted to mmap (i.e.
/// the background fill landed). Used by the warm-search row
/// to wait between the cold cycle and the steady-state
/// timing.
async fn wait_for_mmap_promotion(
    cache: &Arc<DiskCacheStore>,
    uri: SuperfileUri,
    timeout: Duration,
) {
    let start = Instant::now();
    loop {
        // The promotion is observable via `stats().current_bytes`
        // exceeding 0 + a brief yield so the entry swap lands.
        let stats = cache.stats();
        if stats.current_bytes > 0 && stats.n_cold_fetches >= 1 {
            for _ in 0..5 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            // Sanity: re-open and confirm we have a hot entry
            // by asking for the reader — should be near-zero
            // latency (mmap path).
            let _ = cache.reader(&uri).await.expect("warm reader sanity");
            return;
        }
        if start.elapsed() > timeout {
            panic!("cache failed to promote {uri:?} within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ─── Benches ─────────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    // Diagnostic path: short-circuit the normal criterion rows and
    // dump a raw-RTT + cold-fetch-range breakdown to stderr so we
    // can localize where the cold-open / cold-first-search wall
    // time is going. Gated on an env var so the default
    // `cargo bench` path is unaffected.
    if std::env::var("INFINO_DIAG_COLD_PATH").is_ok() {
        diag::diagnose_s3s_fs_cold_path();
        return;
    }

    let rt = Runtime::new().expect("tokio runtime");
    let superfile = superfile_bytes();
    let query = query_vector().to_vec();
    let n = n_docs();
    eprintln!(
        "[object_store_bench] scale: n_docs={n}, dim={}, superfile_size={} MiB",
        crate::corpus::DIM,
        superfile.len() / (1024 * 1024),
    );

    // ── Spawn s3s-fs + upload once. ──────────────────────────────────
    let (_addr, _fs_root, storage, uri) = rt.block_on(setup_s3_fixture(superfile));

    // ── Row 1: cold lazy open via S3. ───────────────────────────────
    // Every iteration: fresh cache, `cache.reader(uri).await`.
    // The background promotion's `tokio::spawn` from the
    // previous iteration is bounded by `tempdir.drop()` so
    // doesn't leak across samples.
    {
        let mut g = c.benchmark_group("object_store_cold_lazy_open");
        g.sample_size(10);
        g.measurement_time(Duration::from_secs(20));

        let storage_for_bench = Arc::clone(&storage);
        g.bench_function(format!("n={n}_s3s_fs"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_for_bench));
                    let t0 = Instant::now();
                    let _reader =
                        rt.block_on(async { cache.reader(&uri).await.expect("cold reader") });
                    total += t0.elapsed();
                    // Drop the cache before the next iteration
                    // — releases the cache root (background
                    // promotion may still be in-flight and may
                    // log errors, but we don't care for the
                    // open-only timing).
                    drop(cache);
                    drop(cache_dir);
                }
                total
            });
        });
        g.finish();
    }

    // ── Row 2: cold lazy open + first vector search. ────────────────
    // The full cold cycle: fresh cache, open, one search.
    // Measures the M2 cold-open + M3 cold-search range
    // budget end-to-end against the S3 wire path.
    {
        let mut g = c.benchmark_group("object_store_cold_first_search");
        g.sample_size(10);
        g.measurement_time(Duration::from_secs(30));

        let storage_for_bench = Arc::clone(&storage);
        let q = query.clone();
        g.bench_function(format!("n={n}_s3s_fs_top{TOP_K}"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_for_bench));
                    let q = q.clone();
                    let t0 = Instant::now();
                    let _hits = rt.block_on(async {
                        let reader = cache.reader(&uri).await.expect("cold reader");
                        let vec = reader.vec().expect("vector reader present");
                        vec.search(
                            "v",
                            &q,
                            TOP_K,
                            DEFAULT_NPROBE,
                            DEFAULT_RERANK_MULT,
                        )
                        .expect("cold vector_search")
                    });
                    total += t0.elapsed();
                    drop(cache);
                    drop(cache_dir);
                }
                total
            });
        });
        g.finish();
    }

    // ── Row 3: cold lazy open + first BM25 search. ──────────────────
    // Same fresh-cache cold cycle as Row 2, but drives the FTS
    // subsection through `FtsReader::open_lazy`: open-time fetch
    // (header + doc-lengths) followed by per-term dict + postings
    // range GETs. Measures the FTS half of the unified cold path
    // against the same S3 wire.
    {
        let mut g = c.benchmark_group("object_store_cold_first_bm25");
        g.sample_size(10);
        g.measurement_time(Duration::from_secs(30));

        let storage_for_bench = Arc::clone(&storage);
        g.bench_function(format!("n={n}_s3s_fs_top{TOP_K}"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_for_bench));
                    let t0 = Instant::now();
                    let _hits = rt.block_on(async {
                        let reader = cache.reader(&uri).await.expect("cold reader");
                        reader
                            .bm25_search(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                            .expect("cold bm25_search")
                    });
                    total += t0.elapsed();
                    drop(cache);
                    drop(cache_dir);
                }
                total
            });
        });
        g.finish();
    }

    // ── Row 4: warm subsequent search (post-promotion). ─────────────
    // Pre-warm the cache once via a cold cycle + wait for
    // the background promotion. Subsequent iterations hit
    // the mmap-backed reader; zero S3 GETs per iteration.
    {
        let (warm_dir, warm_cache) = fresh_cache(Arc::clone(&storage));
        let _ = rt.block_on(async {
            // Trigger cold + wait for promotion.
            let _ = warm_cache.reader(&uri).await.expect("warm prewarm");
            wait_for_mmap_promotion(&warm_cache, uri, Duration::from_secs(60)).await;
        });

        let mut g = c.benchmark_group("object_store_warm_search");
        g.sample_size(50);
        g.measurement_time(Duration::from_secs(10));

        let q = query.clone();
        let cache_ref = Arc::clone(&warm_cache);
        g.bench_function(format!("n={n}_mmap_post_promotion_top{TOP_K}"), |b| {
            b.iter(|| {
                let reader = rt
                    .block_on(async { cache_ref.reader(&uri).await })
                    .expect("warm reader");
                let vec = reader.vec().expect("vector reader present");
                let hits = vec
                    .search(
                        "v",
                        &q,
                        TOP_K,
                        DEFAULT_NPROBE,
                        DEFAULT_RERANK_MULT,
                    )
                    .expect("warm vector_search");
                std::hint::black_box(hits)
            });
        });
        g.finish();

        // Warm BM25 on the *same* promoted cache — the unified
        // segment is fully mmap'd, so the FTS subsection resolves
        // from mmap too (zero S3 GETs), no second promotion wait.
        let mut g = c.benchmark_group("object_store_warm_bm25");
        g.sample_size(50);
        g.measurement_time(Duration::from_secs(10));

        let cache_ref = Arc::clone(&warm_cache);
        g.bench_function(format!("n={n}_mmap_post_promotion_top{TOP_K}"), |b| {
            b.iter(|| {
                let reader = rt
                    .block_on(async { cache_ref.reader(&uri).await })
                    .expect("warm reader");
                let hits = reader
                    .bm25_search(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                    .expect("warm bm25_search");
                std::hint::black_box(hits)
            });
        });
        g.finish();

        drop(warm_cache);
        drop(warm_dir);
    }

    emit_object_store_markdown();
}

// ─── Markdown summary emitter ────────────────────────────────────────

/// Pull criterion's measured `mean.point_estimate` (ns) for
/// each of the cold/warm rows out of
/// `target/criterion/<group>/<bench>/new/estimates.json`,
/// format a single markdown table, and `markdown::emit()`
/// it (stderr unconditionally + README rewrite when
/// `INFINO_BENCH_UPDATE_README=1` is set). The anchor
/// `bench/vector/object_store/cold_warm` matches the
/// `<!-- BEGIN/END -->` markers in `benches/vector/README.md`.
fn emit_object_store_markdown() {
    use crate::markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let n = n_docs();
    let dim = crate::corpus::DIM;
    let superfile_mib = superfile_bytes().len() as f64 / (1024.0 * 1024.0);

    let cold_open_ns = read_mean_ns(
        "object_store_cold_lazy_open",
        &format!("n={n}_s3s_fs"),
    );
    let cold_search_ns = read_mean_ns(
        "object_store_cold_first_search",
        &format!("n={n}_s3s_fs_top{TOP_K}"),
    );
    let cold_bm25_ns = read_mean_ns(
        "object_store_cold_first_bm25",
        &format!("n={n}_s3s_fs_top{TOP_K}"),
    );
    let warm_search_ns = read_mean_ns(
        "object_store_warm_search",
        &format!("n={n}_mmap_post_promotion_top{TOP_K}"),
    );
    let warm_bm25_ns = read_mean_ns(
        "object_store_warm_bm25",
        &format!("n={n}_mmap_post_promotion_top{TOP_K}"),
    );

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile vector + FTS — object-store cold/warm via s3s-fs \
         ({n} docs × dim={dim}, ~{superfile_mib:.0} MiB unified superfile, Sq8 rerank + \
         `title` FTS)\n\n",
    ));
    body.push_str(
        "One unified superfile (vector subsection + FTS subsection in a single \
         Parquet file) served through one `DiskCacheStore`. In-process `s3s-fs` \
         exercises the full SigV4 + HTTP/1.1 path-style range-GET path for \
         request shape; the diagnostic separately reports an adjusted S3 model \
         wall-clock because loopback latency is environment-dependent. \
         `ColdFetchMode::LazyForegroundWithBackgroundFill`: cold foreground \
         returns immediately via `SuperfileReader::open_lazy` (both readers), \
         background downloads the full segment to NVMe + mmaps it, warm calls \
         resolve from mmap (0 S3 GETs).\n\n",
    );
    body.push_str("| Phase | p50 |\n");
    body.push_str("|-------|-----|\n");
    body.push_str(&format!(
        "| Cold open via s3s-fs (Plan 013 M2: footer + per-subsection open-time region) | {} |\n",
        cold_open_ns.map(fmt_time).unwrap_or_else(|| "—".into()),
    ));
    body.push_str(&format!(
        "| Cold first vector search after S3 open (Plan 013 M3: nprobe+1 cluster GETs at nprobe={DEFAULT_NPROBE}) | {} |\n",
        cold_search_ns.map(fmt_time).unwrap_or_else(|| "—".into()),
    ));
    body.push_str(&format!(
        "| Cold first BM25 search after S3 open (`FtsReader::open_lazy`: header + doc-lengths + dict/postings GETs) | {} |\n",
        cold_bm25_ns.map(fmt_time).unwrap_or_else(|| "—".into()),
    ));
    body.push_str(&format!(
        "| Warm subsequent vector search after S3 open (Plan 013 M4: mmap, 0 S3 GETs) | {} |\n",
        warm_search_ns.map(fmt_time).unwrap_or_else(|| "—".into()),
    ));
    body.push_str(&format!(
        "| Warm subsequent BM25 search after S3 open (mmap, 0 S3 GETs) | {} |\n",
        warm_bm25_ns.map(fmt_time).unwrap_or_else(|| "—".into()),
    ));

    crate::markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/object_store/cold_warm".into(),
        body,
    });
}

criterion_group!(benches, bench);

// ─── Diagnostic harness ──────────────────────────────────────────────
//
// Not part of the criterion bench rotation. Bench targets in this
// repo have `harness = false`, so `#[test]` items would be silently
// dropped by the build — a `#[test] #[ignore]` diag would never
// actually run. Instead the diag is a regular module + a regular
// `pub fn diagnose_s3s_fs_cold_path()` which `bench()` invokes
// at the top of its body when `INFINO_DIAG_COLD_PATH=1` is set.
//
// Invocation:
//
//   INFINO_DIAG_COLD_PATH=1 cargo bench --no-default-features \
//     --bench object-store --warm-up-time 1
//
// to localize where cold-path time is going (raw s3s-fs RTT vs.
// our cold-fetch path's range count). When the env var is set,
// `bench()` runs the diagnostic and returns before any of the
// criterion rows fire.

mod diag {
    use super::*;
    use async_trait::async_trait;
    use infino::storage::{ObjectMeta, StorageError};
    use infino::supertable::manifest::SubsectionOffsets;
    use std::ops::Range;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// `StorageProvider` decorator that counts + times every
    /// `head` and `get_range` call. Records each `get_range`'s
    /// `(len_bytes, elapsed_micros)` so we can break down where
    /// per-RTT time is going (small header GETs vs MiB-sized
    /// open-time speculation GETs vs per-cluster block GETs).
    #[derive(Debug)]
    struct CountingStorage {
        inner: Arc<dyn StorageProvider>,
        origin: Instant,
        head_count: AtomicU64,
        head_total_us: AtomicU64,
        range_count: AtomicU64,
        range_total_us: AtomicU64,
        range_log: Mutex<Vec<RequestEvent>>,
    }

    impl CountingStorage {
        fn new(inner: Arc<dyn StorageProvider>) -> Self {
            Self {
                inner,
                origin: Instant::now(),
                head_count: AtomicU64::new(0),
                head_total_us: AtomicU64::new(0),
                range_count: AtomicU64::new(0),
                range_total_us: AtomicU64::new(0),
                range_log: Mutex::new(Vec::new()),
            }
        }

        fn snapshot(&self) -> CountingSnapshot {
            CountingSnapshot {
                head_count: self.head_count.load(Ordering::Relaxed),
                head_total_us: self.head_total_us.load(Ordering::Relaxed),
                range_count: self.range_count.load(Ordering::Relaxed),
                range_total_us: self.range_total_us.load(Ordering::Relaxed),
                range_log: self.range_log.lock().unwrap().clone(),
            }
        }

        fn reset(&self) {
            self.head_count.store(0, Ordering::Relaxed);
            self.head_total_us.store(0, Ordering::Relaxed);
            self.range_count.store(0, Ordering::Relaxed);
            self.range_total_us.store(0, Ordering::Relaxed);
            self.range_log.lock().unwrap().clear();
        }
    }

    #[derive(Debug, Default, Clone)]
    struct RequestEvent {
        len: u64,
        start_us: u128,
        end_us: u128,
    }

    #[derive(Default, Clone)]
    struct CountingSnapshot {
        head_count: u64,
        head_total_us: u64,
        range_count: u64,
        range_total_us: u64,
        range_log: Vec<RequestEvent>,
    }

    impl CountingSnapshot {
        fn diff(&self, prev: &CountingSnapshot) -> CountingSnapshot {
            let log = self.range_log[prev.range_log.len()..].to_vec();
            CountingSnapshot {
                head_count: self.head_count - prev.head_count,
                head_total_us: self.head_total_us - prev.head_total_us,
                range_count: self.range_count - prev.range_count,
                range_total_us: self.range_total_us - prev.range_total_us,
                range_log: log,
            }
        }
    }

    #[async_trait]
    impl StorageProvider for CountingStorage {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            let t0 = Instant::now();
            let r = self.inner.head(uri).await;
            let us = t0.elapsed().as_micros() as u64;
            self.head_count.fetch_add(1, Ordering::Relaxed);
            self.head_total_us.fetch_add(us, Ordering::Relaxed);
            r
        }

        async fn get(&self, uri: &str) -> Result<Bytes, StorageError> {
            self.inner.get(uri).await
        }

        async fn get_range(
            &self,
            uri: &str,
            range: Range<u64>,
        ) -> Result<Bytes, StorageError> {
            let len = range.end - range.start;
            let t0 = Instant::now();
            let start_us = self.origin.elapsed().as_micros();
            let r = self.inner.get_range(uri, range).await;
            let end_us = self.origin.elapsed().as_micros();
            let us = t0.elapsed().as_micros();
            self.range_count.fetch_add(1, Ordering::Relaxed);
            self.range_total_us
                .fetch_add(us as u64, Ordering::Relaxed);
            self.range_log.lock().unwrap().push(RequestEvent {
                len,
                start_us,
                end_us,
            });
            r
        }

        // Plan 013 M5 — must forward to `self.inner.tail` rather
        // than let the trait-default impl call `self.head` +
        // `self.get_range`. The default impl would route through
        // this wrapper's instrumented `head` / `get_range`,
        // splitting one S3 `bytes=-len` suffix-range GET into a
        // (HEAD + bounded GET) pair on the wire and totally
        // erasing the optimization the cold-open path relies on.
        async fn tail(
            &self,
            uri: &str,
            len: u64,
        ) -> Result<(Bytes, u64), StorageError> {
            let t0 = Instant::now();
            let start_us = self.origin.elapsed().as_micros();
            let r = self.inner.tail(uri, len).await;
            let end_us = self.origin.elapsed().as_micros();
            let us = t0.elapsed().as_micros();
            // Count as a single get_range against the wire (which
            // it literally is — one suffix-range GET).
            self.range_count.fetch_add(1, Ordering::Relaxed);
            self.range_total_us
                .fetch_add(us as u64, Ordering::Relaxed);
            // Log with the actual bytes returned so per-range
            // size reporting reflects what came back. On a
            // success the returned `Bytes::len()` is the truth
            // (may be less than `len` if the object is smaller).
            let logged_len = r
                .as_ref()
                .map(|(b, _)| b.len() as u64)
                .unwrap_or(len);
            self.range_log
                .lock()
                .unwrap()
                .push(RequestEvent {
                    len: logged_len,
                    start_us,
                    end_us,
                });
            r
        }

        async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            expected_etag: Option<&str>,
        ) -> Result<(), StorageError> {
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
    }

    fn duration_from_us(us: u128) -> Duration {
        Duration::from_micros(us.min(u64::MAX as u128) as u64)
    }

    fn request_blocking_spans(
        events: &[RequestEvent],
        model: S3LatencyModel,
    ) -> (Duration, Duration, usize) {
        if events.is_empty() {
            return (Duration::ZERO, Duration::ZERO, 0);
        }

        let mut sorted = events.to_vec();
        sorted.sort_unstable_by_key(|e| (e.start_us, e.end_us));

        let mut batches = 0usize;
        let mut raw_blocking = Duration::ZERO;
        let mut model_blocking = Duration::ZERO;

        let mut batch_start = sorted[0].start_us;
        let mut batch_end = sorted[0].end_us;
        let mut batch_model = model.delay_for(sorted[0].len);
        batches += 1;

        for event in sorted.iter().skip(1) {
            if event.start_us <= batch_end {
                batch_end = batch_end.max(event.end_us);
                batch_model = batch_model.max(model.delay_for(event.len));
            } else {
                raw_blocking += duration_from_us(batch_end.saturating_sub(batch_start));
                model_blocking += batch_model;
                batch_start = event.start_us;
                batch_end = event.end_us;
                batch_model = model.delay_for(event.len);
                batches += 1;
            }
        }

        raw_blocking += duration_from_us(batch_end.saturating_sub(batch_start));
        model_blocking += batch_model;
        (raw_blocking, model_blocking, batches)
    }

    fn report(name: &str, snap: &CountingSnapshot, wall: Duration) {
        let head_avg_us = if snap.head_count == 0 {
            0
        } else {
            snap.head_total_us / snap.head_count
        };
        let range_avg_us = if snap.range_count == 0 {
            0
        } else {
            snap.range_total_us / snap.range_count
        };
        let model = S3LatencyModel::from_env_or_default();
        let (raw_blocking, model_blocking, batches) =
            request_blocking_spans(&snap.range_log, model);
        let adjusted_wall = wall.checked_sub(raw_blocking).unwrap_or(Duration::ZERO) + model_blocking;
        eprintln!(
            "[diag] {name}: wall={:>7.1} ms | adjusted_s3_model={:>7.1} ms \
             (ttfb={:>5.1} ms, {:>5.0} MB/s, batches={:>2}, raw_s3s_block={:>7.1} ms, \
             model_block={:>7.1} ms) | HEAD {:>3} calls ({:>5} us avg) | \
             GET_RANGE {:>3} calls ({:>5} us avg, summed {:>7.1} ms)",
            wall.as_secs_f64() * 1e3,
            adjusted_wall.as_secs_f64() * 1e3,
            model.ttfb.as_secs_f64() * 1e3,
            model.bytes_per_sec / 1_000_000.0,
            batches,
            raw_blocking.as_secs_f64() * 1e3,
            model_blocking.as_secs_f64() * 1e3,
            snap.head_count,
            head_avg_us,
            snap.range_count,
            range_avg_us,
            (snap.range_total_us as f64) / 1e3,
        );
        // Range breakdown — log each (len, latency) so we can
        // see e.g. "2 MiB GET took 800ms while 32 B GET took 5ms".
        for (i, event) in snap.range_log.iter().enumerate() {
            let us = event.end_us.saturating_sub(event.start_us);
            let model_us = model.delay_for(event.len).as_micros();
            eprintln!(
                "[diag] {name}:   range[{i:>2}] len={:>10} B  ({:>5.1} KiB)  \
                 raw_lat={:>7} us  model_lat={:>7} us  start={:>10} us  end={:>10} us",
                event.len,
                (event.len as f64) / 1024.0,
                us,
                model_us,
                event.start_us,
                event.end_us,
            );
        }
    }

    /// Probe raw s3s-fs / `S3StorageProvider` round-trip latency
    /// for three range sizes (header-sized, MiB-sized, chunk-sized)
    /// then exercise the cold-open + cold-first-search paths with
    /// the counting wrapper installed so we can see exactly what
    /// the cold-fetch coordinator issues against the wire.
    pub fn diagnose_s3s_fs_cold_path() {
        let rt = Runtime::new().expect("tokio runtime");
        let superfile = superfile_bytes();
        let n = n_docs();
        let query = query_vector().to_vec();

        let (_addr, _fs_root, raw_storage, uri) =
            rt.block_on(setup_s3_fixture(superfile));
        let storage = Arc::new(CountingStorage::new(raw_storage));
        let storage_dyn: Arc<dyn StorageProvider> = Arc::clone(&storage)
            as Arc<dyn StorageProvider>;
        let path = format!("data/seg-{}.sf", uri.0);

        // ── Phase 1: raw RTT probes ─────────────────────────────────
        eprintln!("[diag] === raw S3StorageProvider RTT probes ===");
        for (label, off, len) in [
            ("32B_head", 0u64, 32u64),
            ("64KiB_mid", 1024 * 1024, 64 * 1024),
            ("2MiB_open_spec", 0, 2 * 1024 * 1024),
            ("4MiB_chunk", 0, 4 * 1024 * 1024),
        ] {
            let len = len.min(superfile.len() as u64 - off);
            let mut total = Duration::ZERO;
            const ITERS: u32 = 5;
            for _ in 0..ITERS {
                let t0 = Instant::now();
                let _b = rt
                    .block_on(storage_dyn.get_range(&path, off..off + len))
                    .expect("raw range");
                total += t0.elapsed();
            }
            eprintln!(
                "[diag] raw_get_range[{label:<14}] len={:>8} B  avg={:>6.2} ms over {ITERS} iters",
                len,
                total.as_secs_f64() / ITERS as f64 * 1e3,
            );
        }
        storage.reset();

        // Build the SubsectionOffsets the manifest would carry
        // post-Plan-013 M6, so we can A/B the cold-open path
        // unhinted (pre-M6, 2-RTT sequential) vs hinted (M6,
        // 1-RTT parallel prefetch).
        let offsets = build_offsets_from_bytes(superfile);
        eprintln!(
            "[diag] manifest hints: total={} B  vec={:?}  fts={:?}",
            offsets.total_size, offsets.vec, offsets.fts
        );

        // ── Phase 2a: cold-open UNHINTED (pre-M6, 2 RTTs) ───────────
        eprintln!("[diag] === cold-open UNHINTED via cache.reader (3 fresh-cache iters) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let t0 = Instant::now();
            let _reader = rt
                .block_on(cache.reader(&uri))
                .expect("cold reader");
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_open_unhinted[{i}]"), &snap, wall);
            // Let the previous iter's bg fill stop touching s3s-fs
            // before the next cold timing starts, so contention
            // doesn't poison the measurement. The `sleep` itself
            // must be `await`ed inside `block_on` so it enters
            // the runtime context.
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 2b: cold-open HINTED (M6, 1 RTT parallel) ─────────
        eprintln!(
            "[diag] === cold-open HINTED via cache.reader_with_hints (3 fresh-cache iters) ==="
        );
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let off_ref = offsets;
            let t0 = Instant::now();
            let _reader = rt
                .block_on(cache.reader_with_hints(&uri, Some(&off_ref)))
                .expect("cold reader");
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_open_hinted[{i}]"), &snap, wall);
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 3a: cold first search UNHINTED ────────────────────
        eprintln!(
            "[diag] === cold first search UNHINTED (nprobe={DEFAULT_NPROBE}, top={TOP_K}) ==="
        );
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let q = query.clone();
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache.reader(&uri).await.expect("cold reader");
                let vec = reader.vec().expect("vector reader present");
                vec.search("v", &q, TOP_K, DEFAULT_NPROBE, DEFAULT_RERANK_MULT)
                    .expect("cold vector_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_first_search_unhinted[{i}]"), &snap, wall);
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 3b: cold first search HINTED ──────────────────────
        eprintln!(
            "[diag] === cold first search HINTED (nprobe={DEFAULT_NPROBE}, top={TOP_K}) ==="
        );
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let q = query.clone();
            let off_ref = offsets;
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache
                    .reader_with_hints(&uri, Some(&off_ref))
                    .await
                    .expect("cold reader");
                let vec = reader.vec().expect("vector reader present");
                vec.search("v", &q, TOP_K, DEFAULT_NPROBE, DEFAULT_RERANK_MULT)
                    .expect("cold vector_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_first_search_hinted[{i}]"), &snap, wall);
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        eprintln!("[diag] === scale: n_docs={n}, superfile_size={} MiB ===", superfile.len() / (1024 * 1024));
    }

    /// Synthesize the [`SubsectionOffsets`] the writer would have
    /// emitted on commit, by parsing the parquet KV metadata out
    /// of the freshly-built superfile bytes. Mirrors the
    /// `build_subsection_offsets` helper in the writer.
    fn build_offsets_from_bytes(bytes: &[u8]) -> SubsectionOffsets {
        use infino::superfile::format::{footer::read_kv_metadata, kv};
        let kvs = read_kv_metadata(bytes).expect("read_kv_metadata");
        let get = |k: &str| -> Option<u64> { kvs.get(k).and_then(|s| s.parse::<u64>().ok()) };
        let vec = match (get(kv::VEC_OFFSET), get(kv::VEC_LENGTH)) {
            (Some(o), Some(l)) if l > 0 => Some((o, l)),
            _ => None,
        };
        let fts = match (get(kv::FTS_OFFSET), get(kv::FTS_LENGTH)) {
            (Some(o), Some(l)) if l > 0 => Some((o, l)),
            _ => None,
        };
        SubsectionOffsets {
            total_size: bytes.len() as u64,
            vec,
            fts,
        }
    }
}
