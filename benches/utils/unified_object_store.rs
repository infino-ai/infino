// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! **Quick-iteration harness** for the object-store cold-fetch path.
//!
//! Fast dev-loop probe on a single superfile over `s3s-fs` (default 100k docs;
//! `INFINO_BENCH_FULL=1` → 1M). Canonical tiered benchmarks (superfile 1M /
//! supertable 10M × warm/warm/cold) live in `vector_*` / `fts_*` via `tiers.rs`.
//! Use this bench to iterate on request shape and diagnostics, not headline SLA rows.
//!
//! Exercises unified vector + FTS cold-open / cold-first-search
//! / warm-search against an in-process S3 server (`s3s-fs`).
//!
//! Backend selection here is intentionally **S3-only and inference-based**
//! (`s3s-fs`, or real S3 when `INFINO_REAL_S3_BUCKET` is set) — it does not
//! use the `INFINO_BENCH_STORE` selector or the Azure backend that the
//! canonical benches do. This is a diagnostics probe, not a headline bench,
//! so it keeps its own minimal fixture rather than routing through
//! `tiers::Backend`.
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
//!    empty cache; pays the cold-open budget (Parquet
//!    footer + per-subsection open-time-region GETs). One open
//!    serves both the vector and FTS readers.
//! 2. **Cold first vector search after S3 open** — `vec.search`
//!    at the default `(nprobe, rerank_mult)` against a freshly
//!    opened reader and empty superfile-data cache; pays the
//!    cold-search budget (~nprobe + 1 cluster GETs), excluding
//!    file open.
//! 3. **Cold first BM25 search after S3 open** — `bm25_search`
//!    against a freshly opened reader and empty superfile-data cache;
//!    pays the FTS lazy open-time fetch (header + doc-lengths) plus
//!    per-term dict/postings range GETs (`FtsReader::open_lazy`
//!    mirroring the vector path), excluding file open.
//! 4. **Warm subsequent search after S3 open** — after the
//!    background promotion completes, the cache returns the
//!    mmap-backed reader and both vector + BM25 searches
//!    resolve entirely from mmap (zero S3 GETs).
//!
//! ## Invocation
//!
//! ```text
//! cargo bench -- object-store
//! INFINO_REAL_S3_BUCKET=<bucket> cargo bench -- object-store
//! INFINO_BENCH_UPDATE_README=1 cargo bench -- object-store
//! ```
//!
//! Scale is fixed by shape at [`corpus::SUPERFILE_DOCS`] (1M × 384,
//! ~1.5 GiB) — this is the superfile warm/cold tier and matches the
//! superfile warm benches. There is no `INFINO_BENCH_FULL` knob. The
//! Criterion rows run over the in-process `s3s-fs` server by default;
//! setting `INFINO_REAL_S3_BUCKET` (or `INFINO_TEST_REAL_S3_BUCKET`)
//! reruns the same rows against real AWS S3.
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
//!     validating request shape: GET count, byte ranges, and
//!     overlap. Its loopback latency is not treated as S3 latency;
//!     the diagnostic prints an adjusted model line that replaces
//!     the observed s3s-fs blocking span with a configurable S3
//!     TTFB + throughput model.

#![allow(clippy::too_many_arguments)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use arrow_array::{
    Array, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::config::{
    CompactionSettings, Config, StorageBackend, StorageColdFetchMode, StorageSettings,
    SupertableSettings,
};
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::SuperfileUri;
use infino::supertable::query::VectorSearchOptions;
use infino::supertable::reader_cache::DiskCacheStore;
use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;
use s3s::auth::SimpleAuth;
use s3s::service::S3ServiceBuilder;
use s3s_fs::FileSystem;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

// ─── Constants ───────────────────────────────────────────────────────

const TEST_BUCKET: &str = "infino-objstore-bench";
const TEST_REGION: &str = "us-east-1";
const TEST_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const TEST_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

const QUICK_ITER_DEFAULT_DOCS: usize = 100_000;
const REAL_S3_MAX_ITERS: u64 = 3;

/// Doc count for this quick-iter harness only (`INFINO_BENCH_FULL=1` → 1M).
fn quick_iter_n_docs() -> usize {
    if std::env::var("INFINO_BENCH_FULL").is_ok() {
        crate::corpus::SUPERFILE_DOCS
    } else {
        QUICK_ITER_DEFAULT_DOCS
    }
}

/// Default `(nprobe, rerank_mult)` for the search rows.
/// Matches the production default in `VectorSearchOptions`.
const DEFAULT_NPROBE: usize = 8;
const DEFAULT_RERANK_MULT: usize = 20;
const TOP_K: usize = 10;

const BENCH_NPROBE: usize = DEFAULT_NPROBE;

/// Primary-key column. `SuperfileBuilder` requires the id column
/// to be `Decimal128(38, 0)` (the supertable's snowflake id type).
const ID_COLUMN: &str = "doc_id";
/// Vector column logical name (lives only in the embedded vector
/// blob, not the Parquet schema).
const VEC_COLUMN: &str = "v";
/// FTS column registered on the unified fixture. Single `title`
/// column — same shape `benches/utils/superfile.rs` (`fts`) builds.
/// It stays in the Parquet body (SQL-visible) *and* is indexed
/// into the FTS blob, which is the whole point of the unified
/// layout.
const FTS_COLUMN: &str = "title";
/// Zipfian-common term (`MmapTextCorpus` plants `term00001` as
/// the highest-frequency term), so the cold BM25 row exercises a
/// real multi-block postings fetch rather than a df=1 sliver.
const FTS_QUERY_TERM: &str = "term00001";
const FTS_MULTI_QUERY: &str = "term00001 term00002 term00003";

/// Bytes per mebibyte (integer + float forms) for size logging.
const BYTES_PER_MIB: usize = 1024 * 1024;
const BYTES_PER_MIB_F64: f64 = 1024.0 * 1024.0;
/// RNG seed for the vector corpus fixtures.
const CORPUS_SEED: u64 = 1;
/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Random-rotation RNG seed for the bench vector index.
const ROT_SEED: u64 = 7;
/// Seconds-to-milliseconds factor for diagnostic latency output.
const MS_PER_SEC: f64 = 1e3;
/// Microseconds-to-milliseconds divisor for range-timer output.
const US_PER_MS: f64 = 1e3;

// ─── Fixtures (built once per `cargo bench` invocation) ──────────────

static SUPERFILE_BYTES: OnceLock<Bytes> = OnceLock::new();
static QUERY_VECTOR: OnceLock<Vec<f32>> = OnceLock::new();

fn superfile_bytes() -> &'static Bytes {
    SUPERFILE_BYTES.get_or_init(build_superfile_bytes)
}

fn query_vector() -> &'static [f32] {
    QUERY_VECTOR
        .get_or_init(|| {
            let n = quick_iter_n_docs();
            let v = crate::corpus::MmapVectorCorpus::generate(
                n,
                crate::corpus::n_cent(n),
                CORPUS_SEED,
                true,
            );
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
    let n = quick_iter_n_docs();
    let n_cent = crate::corpus::n_cent(n);
    let dim = crate::corpus::DIM;

    let vectors_mmap = crate::corpus::MmapVectorCorpus::generate(n, n_cent, CORPUS_SEED, true);
    let vectors = vectors_mmap.as_slice();
    let text = crate::corpus::MmapTextCorpus::generate(n, 1);

    // Schema = id (Decimal128, as the supertable injects) + the FTS
    // text column. The vector column is a logical name only; its f32
    // buffer is passed alongside each batch, not as a schema field.
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            ID_COLUMN,
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
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
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
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
            .with_precision_and_scale(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE)
            .expect("decimal128 with_precision_and_scale");
        let titles = LargeStringArray::from(
            (start..start + len)
                .map(|i| text.doc(i))
                .collect::<Vec<_>>(),
        );
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
        bytes.len() / BYTES_PER_MIB,
        t0.elapsed().as_secs_f32(),
    );
    Bytes::from(bytes)
}

// ─── S3 latency model (adjusted diagnostic) ────────────
//
// `s3s-fs` over loopback faithfully reproduces the S3 *request
// count* and *byte volume* (the things GET-minimization
// optimizes), but not S3's *wall-clock* — its per-request latency
// is environment-dependent and unrelated to real S3. To get a
// meaningful cold-open / cold-search wall-clock signal while
// iterating, the diagnostic reports a synthetic
// AWS-S3-in-region timing model on top of the real request shape:
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
//
//   INFINO_S3_COST_HEAD_PER_1000=<f64>  request cost for HEAD calls
//                                      (default $0.0004 / 1K)
//   INFINO_S3_COST_GET_PER_1000=<f64>   request cost for GET/range calls
//                                      (default $0.0004 / 1K)
//   INFINO_S3_COST_DATA_PER_GIB=<f64>   optional transfer/read byte cost
//                                      (default $0.0 / GiB; same-region
//                                      S3→EC2 transfer is usually free)
#[derive(Debug, Clone, Copy)]
struct S3LatencyModel {
    ttfb: Duration,
    bytes_per_sec: f64,
}

impl S3LatencyModel {
    /// Read the model used for adjusted diagnostic reporting.
    /// This never changes the measured code path.
    fn from_env_or_default() -> Self {
        const TTFB_MS: f64 = 100.0;
        const MBPS: f64 = 100.0;
        Self {
            ttfb: Duration::from_secs_f64(TTFB_MS / 1000.0),
            bytes_per_sec: MBPS * 1_000_000.0,
        }
    }

    fn delay_for(&self, bytes: u64) -> Duration {
        self.ttfb + Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec)
    }
}

#[derive(Debug, Clone, Copy)]
struct S3CostModel {
    head_per_1000: f64,
    get_per_1000: f64,
    data_per_gib: f64,
}

#[derive(Debug, Clone, Copy)]
struct S3CostBreakdown {
    request_usd: f64,
    data_usd: f64,
}

impl S3CostBreakdown {
    fn total_usd(self) -> f64 {
        self.request_usd + self.data_usd
    }
}

impl S3CostModel {
    fn from_env_or_default() -> Self {
        Self {
            head_per_1000: 0.0004,
            get_per_1000: 0.0004,
            data_per_gib: 0.0,
        }
    }

    fn request_cost(&self, head_count: u64, get_count: u64) -> f64 {
        (head_count as f64 * self.head_per_1000 + get_count as f64 * self.get_per_1000) / 1000.0
    }

    fn data_cost(&self, bytes: u64) -> f64 {
        let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        gib * self.data_per_gib
    }

    fn cost_for(&self, head_count: u64, get_count: u64, bytes: u64) -> S3CostBreakdown {
        S3CostBreakdown {
            request_usd: self.request_cost(head_count, get_count),
            data_usd: self.data_cost(bytes),
        }
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
    let path = uri.storage_path();
    // Upload against the raw provider — fixture-setup latency is
    // not part of the measured cold path.
    storage
        .put_atomic(&path, superfile.clone())
        .await
        .expect("upload superfile to s3s-fs");
    eprintln!("[object_store_bench] s3s-fs spawned on {endpoint}, superfile uploaded to {path}");
    (addr, fs_root, storage, uri)
}

struct BenchFixture {
    storage: Arc<dyn StorageProvider>,
    uri: SuperfileUri,
    storage_label: &'static str,
    real_s3: bool,
    cleanup_path: Option<String>,
    _fs_root: Option<TempDir>,
}

impl BenchFixture {
    async fn cleanup(&self) {
        if let Some(path) = &self.cleanup_path {
            let result = self.storage.delete(path).await;
            eprintln!("[object_store_bench] cleanup {path}: {result:?}");
        }
    }
}

fn real_s3_bucket_env() -> Option<String> {
    crate::tiers::real_s3_bucket_env()
}

fn real_s3_prefix_root_env() -> String {
    crate::tiers::real_s3_prefix_root("infino-real-s3-bench")
}

fn unique_bench_prefix(root: &str) -> String {
    let unique = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos()
    );
    format!("{}/{}", root.trim_matches('/'), unique)
}

async fn setup_bench_fixture(superfile: &Bytes) -> BenchFixture {
    if let Some(bucket) = real_s3_bucket_env() {
        let prefix = unique_bench_prefix(&real_s3_prefix_root_env());
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_prefix(&bucket, &prefix)
                .expect("real S3 benchmark provider"),
        );
        let uri = SuperfileUri::new_v4();
        let path = uri.storage_path();
        storage
            .put_atomic(&path, superfile.clone())
            .await
            .expect("upload superfile to real S3");
        eprintln!(
            "[object_store_bench] real S3 fixture uploaded: bucket={bucket} prefix={prefix} path={path}"
        );
        BenchFixture {
            storage,
            uri,
            storage_label: "real_s3",
            real_s3: true,
            cleanup_path: Some(path),
            _fs_root: None,
        }
    } else {
        let (_addr, fs_root, storage, uri) = setup_s3_fixture(superfile).await;
        BenchFixture {
            storage,
            uri,
            storage_label: "s3s_fs",
            real_s3: false,
            cleanup_path: None,
            _fs_root: Some(fs_root),
        }
    }
}

/// Fresh disk-cache in `LazyForegroundWithBackgroundFill` mode.
/// Returns the cache + its temp root (drop after to GC).
fn fresh_cache(storage: Arc<dyn StorageProvider>) -> (TempDir, Arc<DiskCacheStore>) {
    crate::tiers::fresh_superfile_cache(storage)
}

// ─── Benches ─────────────────────────────────────────────────────────

/// Cold-row iteration count. Bounded for real S3 (each iter is a real
/// network round trip + a fresh-cache cold open), more samples on s3s-fs
/// for a stabler p50.
fn cold_iters(real_s3: bool) -> usize {
    if real_s3 {
        REAL_S3_MAX_ITERS as usize
    } else {
        10
    }
}

pub fn run() {
    // Deeper opt-in diagnostics (separate fixtures + breakdowns) stay
    // gated. The cold-path stats themselves are always shown below.
    if std::env::var("INFINO_DIAG_REAL_S3").is_ok() {
        diag::diagnose_real_s3_cold_path();
        return;
    }
    if std::env::var("INFINO_DIAG_REAL_S3_SUPERTABLE").is_ok() {
        diag::diagnose_real_s3_supertable_e2e();
        return;
    }
    if std::env::var("INFINO_DIAG_QUERY_SQL_OVERHEAD").is_ok() {
        diag::diagnose_query_sql_overhead();
        return;
    }

    let rt = Runtime::new().expect("tokio runtime");
    let superfile = superfile_bytes();
    let query = query_vector().to_vec();
    let n = quick_iter_n_docs();
    let nprobe = BENCH_NPROBE;
    eprintln!(
        "[object_store_bench] scale: n_docs={n}, dim={}, superfile_size={} MiB",
        crate::corpus::DIM,
        superfile.len() / BYTES_PER_MIB,
    );

    // Upload once to the selected object-store backend. Default remains
    // s3s-fs; set INFINO_REAL_S3_BUCKET to run the same rows against
    // actual AWS S3.
    let fixture = rt.block_on(setup_bench_fixture(superfile));
    let storage = Arc::clone(&fixture.storage);
    let uri = fixture.uri;
    let storage_label = fixture.storage_label;
    let real_s3 = fixture.real_s3;
    let iters = cold_iters(real_s3);

    // Each phase runs through a request-counting store: it prints the
    // detailed per-iter `[diag]` line (modeled S3 latency + estimated cost
    // for s3s-fs; true wall-clock + cost for real S3) and returns the p50
    // latency + median cost for the summary table. The RSS sampler bounds
    // the whole phase.
    let measure = |name: &str, op: diag::ColdOp| {
        let sampler = crate::rss::PeakSampler::start_default();
        let phase = diag::measure_cold_phase(
            &rt,
            Arc::clone(&storage),
            &uri,
            &query,
            nprobe,
            real_s3,
            name,
            op,
            iters,
        );
        (phase, sampler.stop_stats())
    };

    let open = measure("cold_open", diag::ColdOp::Open);
    let vec = measure("cold_first_vector_search", diag::ColdOp::VectorSearch);
    let bm25 = measure("cold_first_bm25_search", diag::ColdOp::Bm25Search);

    rt.block_on(fixture.cleanup());
    emit_object_store(storage_label, real_s3, n, open, vec, bm25);
}

// ─── Report emitter ──────────────────────────────────────────────────

/// Emit the three measured cold rows through the custom report harness:
/// terminal table + run-to-run deltas, plus (when
/// `INFINO_BENCH_UPDATE_README=1`) the `bench/vector/object_store/cold`
/// README anchor. The `p50` column is the modeled S3 latency for s3s-fs
/// (loopback removed, TTFB+throughput added back) or the true wall-clock
/// for real S3; `Est S3 cost` is the modeled per-op request+data cost.
fn emit_object_store(
    storage_label: &str,
    real_s3: bool,
    n: usize,
    cold_open: (diag::ColdPhase, crate::rss::RssStats),
    cold_vec: (diag::ColdPhase, crate::rss::RssStats),
    cold_bm25: (diag::ColdPhase, crate::rss::RssStats),
) {
    use crate::markdown::fmt_time;
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};

    let dim = crate::corpus::DIM;
    let superfile_mib = superfile_bytes().len() as f64 / BYTES_PER_MIB_F64;

    let rss_cells = |stats: crate::rss::RssStats| -> Vec<Cell> {
        vec![
            metric(
                stats.peak_rss_bytes as f64,
                crate::rss::fmt_bytes(stats.peak_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.median_rss_bytes as f64,
                crate::rss::fmt_bytes(stats.median_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.p90_rss_bytes as f64,
                crate::rss::fmt_bytes(stats.p90_rss_bytes),
                Better::Lower,
            ),
        ]
    };

    let row = |label: &str, m: (diag::ColdPhase, crate::rss::RssStats)| -> Vec<Cell> {
        let ns = m.0.p50.as_secs_f64() * 1e9;
        let mut cells = vec![
            text(label),
            metric(ns, fmt_time(ns), Better::Lower),
            metric(m.0.cost_usd, format!("${:.6}", m.0.cost_usd), Better::Lower),
        ];
        cells.extend(rss_cells(m.1));
        cells
    };

    let latency_col = if real_s3 {
        "p50 (real S3)"
    } else {
        "p50 (modeled S3)"
    };
    let backend_note = if real_s3 {
        "Latency is the true real-S3 wall-clock; cost is the modeled per-op request + data cost."
    } else {
        "Latency is the modeled S3 wall-clock (the s3s-fs loopback blocking span is subtracted and \
         a TTFB + throughput model added back per overlap-coalesced GET batch — the raw s3s-fs \
         loopback time is environment-dependent and not representative); cost is the modeled per-op \
         request + data cost. Set INFINO_REAL_S3_BUCKET to measure real S3 directly."
    };

    let mut report = Report::load("object-store");
    report.emit(&Section {
        anchor: "bench/vector/object_store/cold".into(),
        title: format!(
            "Superfile vector + FTS — object-store cold ({storage_label}, {n} docs × dim={dim}, ~{superfile_mib:.0} MiB unified superfile, Sq8 rerank + `title` FTS)"
        ),
        note: format!(
            "One unified superfile (vector subsection + FTS subsection in a single Parquet file) \
             served through one `DiskCacheStore` in `LazyForegroundWithBackgroundFill` mode. \
             In-process `s3s-fs` exercises the full SigV4 + HTTP/1.1 path-style range-GET path for \
             request shape. {backend_note} The detailed per-range `[diag]` breakdown prints above \
             this table. p50 over repeated fresh-cache runs. Δ is vs the previous run."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Phase".into(),
                latency_col.into(),
                "Est S3 cost".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows: vec![
                row("cold open (footer + per-subsection open-time region)", cold_open),
                row(
                    &format!("cold first vector_search (nprobe+1 cluster GETs at nprobe={DEFAULT_NPROBE})"),
                    cold_vec,
                ),
                row(
                    "cold first bm25_search (FtsReader::open_lazy: header + doc-lengths + dict/postings GETs)",
                    cold_bm25,
                ),
            ],
        }],
    });
    report.save();
}

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
//   INFINO_DIAG_COLD_PATH=1 cargo bench -- object-store
//
// to localize where cold-path time is going (raw s3s-fs RTT vs.
// our cold-fetch path's range count). When the env var is set,
// `bench()` runs the diagnostic and returns before any of the
// criterion rows fire.

pub(crate) mod diag {
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

        fn range_bytes(&self) -> u64 {
            self.range_log.iter().map(|e| e.len).sum()
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

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            self.inner.get(uri).await
        }

        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            let len = range.end - range.start;
            let t0 = Instant::now();
            let start_us = self.origin.elapsed().as_micros();
            let r = self.inner.get_range(uri, range).await;
            let end_us = self.origin.elapsed().as_micros();
            let us = t0.elapsed().as_micros();
            self.range_count.fetch_add(1, Ordering::Relaxed);
            self.range_total_us.fetch_add(us as u64, Ordering::Relaxed);
            self.range_log.lock().unwrap().push(RequestEvent {
                len,
                start_us,
                end_us,
            });
            r
        }

        // Must forward to `self.inner.tail` rather than let the
        // trait-default impl call `self.head` + `self.get_range`.
        // The default impl would route through this wrapper's
        // instrumented `head` / `get_range`, splitting one S3
        // `bytes=-len` suffix-range GET into a
        // (HEAD + bounded GET) pair on the wire and totally
        // erasing the optimization the cold-open path relies on.
        async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
            let t0 = Instant::now();
            let start_us = self.origin.elapsed().as_micros();
            let r = self.inner.tail(uri, len).await;
            let end_us = self.origin.elapsed().as_micros();
            let us = t0.elapsed().as_micros();
            // Count as a single get_range against the wire (which
            // it literally is — one suffix-range GET).
            self.range_count.fetch_add(1, Ordering::Relaxed);
            self.range_total_us.fetch_add(us as u64, Ordering::Relaxed);
            // Log with the actual bytes returned so per-range
            // size reporting reflects what came back. On a
            // success the returned `Bytes::len()` is the truth
            // (may be less than `len` if the object is smaller).
            let logged_len = r.as_ref().map(|(b, _)| b.len() as u64).unwrap_or(len);
            self.range_log.lock().unwrap().push(RequestEvent {
                len: logged_len,
                start_us,
                end_us,
            });
            r
        }

        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
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

    fn report(name: &str, snap: &CountingSnapshot, wall: Duration, real_s3: bool) {
        let head_avg_us = snap.head_total_us.checked_div(snap.head_count).unwrap_or(0);
        let range_avg_us = snap
            .range_total_us
            .checked_div(snap.range_count)
            .unwrap_or(0);
        let model = S3LatencyModel::from_env_or_default();
        let cost_model = S3CostModel::from_env_or_default();
        let (raw_blocking, model_blocking, batches) =
            request_blocking_spans(&snap.range_log, model);
        let adjusted_wall =
            wall.checked_sub(raw_blocking).unwrap_or(Duration::ZERO) + model_blocking;
        let bg_chunk_min = std::env::var("INFINO_DIAG_BG_CHUNK_MIN_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(8 * 1024 * 1024);
        let foreground_events: Vec<RequestEvent> = snap
            .range_log
            .iter()
            .filter(|e| e.len < bg_chunk_min)
            .cloned()
            .collect();
        let background_fill_events = snap.range_log.len() - foreground_events.len();
        let (fg_raw_blocking, fg_model_blocking, fg_batches) =
            request_blocking_spans(&foreground_events, model);
        let adjusted_foreground_wall =
            wall.checked_sub(fg_raw_blocking).unwrap_or(Duration::ZERO) + fg_model_blocking;
        let range_bytes = snap.range_bytes();
        let cost = cost_model.cost_for(snap.head_count, snap.range_count, range_bytes);
        let (returned_gets, after_return_gets) = if snap.range_log.is_empty() {
            (0usize, 0usize)
        } else {
            let phase_start_us = snap
                .range_log
                .iter()
                .map(|e| e.start_us)
                .min()
                .unwrap_or_default();
            let return_us = phase_start_us + wall.as_micros();
            let returned = snap
                .range_log
                .iter()
                .filter(|e| e.end_us <= return_us)
                .count();
            (returned, snap.range_log.len() - returned)
        };
        if real_s3 {
            // Real AWS S3: every latency below is a true wire
            // measurement. We deliberately omit the synthetic
            // `S3LatencyModel` projection (ttfb/mbps/
            // adjusted_s3_model) — that model exists only to
            // estimate S3 latency from the in-process s3s-fs
            // path and is meaningless when the wall clock IS
            // real S3. `raw_block` is the real time spent
            // blocked on overlap-coalesced GET batches; the
            // `$` cost projection is kept because dollar cost
            // is not observable from a latency measurement.
            eprintln!(
                "[diag] {name}: wall={:>7.1} ms (real S3, no synthetic latency model) | \
                 raw_block={:>7.1} ms over {:>2} batch(es) | HEAD {:>3} calls ({:>5} us avg) | \
                 GET_RANGE {:>3} calls ({:>5} us avg, summed {:>7.1} ms, \
                 returned={} after_return={}) | \
                 bytes={:>10} B ({:>7.2} MiB) | s3_cost=${:.9} \
                 (requests=${:.9}, data=${:.9})",
                wall.as_secs_f64() * MS_PER_SEC,
                raw_blocking.as_secs_f64() * MS_PER_SEC,
                batches,
                snap.head_count,
                head_avg_us,
                snap.range_count,
                range_avg_us,
                (snap.range_total_us as f64) / US_PER_MS,
                returned_gets,
                after_return_gets,
                range_bytes,
                range_bytes as f64 / 1024.0 / 1024.0,
                cost.total_usd(),
                cost.request_usd,
                cost.data_usd,
            );
            if background_fill_events > 0 {
                eprintln!(
                    "[diag] {name}:   foreground_only_excluding_cache_fill_chunks(>={} B): \
                     raw_block={:>7.1} ms over {:>2} batch(es) (bg_fill_gets={})",
                    bg_chunk_min,
                    fg_raw_blocking.as_secs_f64() * MS_PER_SEC,
                    fg_batches,
                    background_fill_events,
                );
            }
        } else {
            eprintln!(
                "[diag] {name}: wall={:>7.1} ms | adjusted_s3_model={:>7.1} ms \
                 (ttfb={:>5.1} ms, {:>5.0} MB/s, batches={:>2}, raw_s3s_block={:>7.1} ms, \
                 model_block={:>7.1} ms) | HEAD {:>3} calls ({:>5} us avg) | \
                 GET_RANGE {:>3} calls ({:>5} us avg, summed {:>7.1} ms, \
                 returned={} after_return={}) | \
                 bytes={:>10} B ({:>7.2} MiB) | s3_cost=${:.9} \
                 (requests=${:.9}, data=${:.9})",
                wall.as_secs_f64() * MS_PER_SEC,
                adjusted_wall.as_secs_f64() * MS_PER_SEC,
                model.ttfb.as_secs_f64() * MS_PER_SEC,
                model.bytes_per_sec / 1_000_000.0,
                batches,
                raw_blocking.as_secs_f64() * MS_PER_SEC,
                model_blocking.as_secs_f64() * MS_PER_SEC,
                snap.head_count,
                head_avg_us,
                snap.range_count,
                range_avg_us,
                (snap.range_total_us as f64) / US_PER_MS,
                returned_gets,
                after_return_gets,
                range_bytes,
                range_bytes as f64 / 1024.0 / 1024.0,
                cost.total_usd(),
                cost.request_usd,
                cost.data_usd,
            );
            if background_fill_events > 0 {
                eprintln!(
                    "[diag] {name}:   foreground_estimate_excluding_cache_fill_chunks(>={} B): \
                     adjusted_s3_model={:>7.1} ms (fg_batches={:>2}, fg_raw_s3s_block={:>7.1} ms, \
                     fg_model_block={:>7.1} ms, bg_fill_gets={})",
                    bg_chunk_min,
                    adjusted_foreground_wall.as_secs_f64() * MS_PER_SEC,
                    fg_batches,
                    fg_raw_blocking.as_secs_f64() * MS_PER_SEC,
                    fg_model_blocking.as_secs_f64() * MS_PER_SEC,
                    background_fill_events,
                );
            }
        }
        eprintln!(
            "[diag] {name}:   cost_model HEAD=${:.7}/1K GET=${:.7}/1K DATA=${:.4}/GiB",
            cost_model.head_per_1000, cost_model.get_per_1000, cost_model.data_per_gib,
        );
        // Range breakdown — log each (len, latency) so we can
        // see e.g. "2 MiB GET took 800ms while 32 B GET took 5ms".
        for (i, event) in snap.range_log.iter().enumerate() {
            let us = event.end_us.saturating_sub(event.start_us);
            let event_cost = cost_model.cost_for(0, 1, event.len);
            if real_s3 {
                eprintln!(
                    "[diag] {name}:   range[{i:>2}] len={:>10} B  ({:>5.1} KiB)  \
                     raw_lat={:>7} us  cost=${:.9}  \
                     start={:>10} us  end={:>10} us",
                    event.len,
                    (event.len as f64) / 1024.0,
                    us,
                    event_cost.total_usd(),
                    event.start_us,
                    event.end_us,
                );
            } else {
                let model_us = model.delay_for(event.len).as_micros();
                eprintln!(
                    "[diag] {name}:   range[{i:>2}] len={:>10} B  ({:>5.1} KiB)  \
                     raw_lat={:>7} us  model_lat={:>7} us  cost=${:.9}  \
                     start={:>10} us  end={:>10} us",
                    event.len,
                    (event.len as f64) / 1024.0,
                    us,
                    model_us,
                    event_cost.total_usd(),
                    event.start_us,
                    event.end_us,
                );
            }
        }
    }

    /// One cold phase's aggregate for the default report table.
    pub(super) struct ColdPhase {
        /// p50 latency — the modeled S3 wall-clock for s3s-fs (loopback
        /// blocking subtracted, TTFB+throughput model added back), or the
        /// true wall-clock for real S3.
        pub(super) p50: Duration,
        /// Median estimated S3 cost (USD) of one cold operation.
        pub(super) cost_usd: f64,
    }

    /// Which cold operation a phase drives.
    pub(super) enum ColdOp {
        Open,
        VectorSearch,
        Bm25Search,
    }

    /// Run `iters` fresh-cache cold operations through a request-counting
    /// store. Prints the per-iter `[diag]` breakdown (modeled S3 latency +
    /// estimated cost for s3s-fs; true wall-clock + cost for real S3) via
    /// [`report`], and returns the p50 (modeled/real) latency + median
    /// estimated cost for the summary table. `raw_storage` is the
    /// un-instrumented backend.
    pub(super) fn measure_cold_phase(
        rt: &Runtime,
        raw_storage: Arc<dyn StorageProvider>,
        uri: &SuperfileUri,
        query: &[f32],
        nprobe: usize,
        real_s3: bool,
        name: &str,
        op: ColdOp,
        iters: usize,
    ) -> ColdPhase {
        let storage = Arc::new(CountingStorage::new(raw_storage));
        let storage_dyn: Arc<dyn StorageProvider> =
            Arc::clone(&storage) as Arc<dyn StorageProvider>;
        let model = S3LatencyModel::from_env_or_default();
        let cost_model = S3CostModel::from_env_or_default();
        let mut latencies = Vec::with_capacity(iters);
        let mut costs = Vec::with_capacity(iters);
        for i in 0..iters {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let t0 = Instant::now();
            match op {
                ColdOp::Open => {
                    let r = rt.block_on(async { cache.reader(uri).await.expect("cold reader") });
                    std::hint::black_box(r);
                }
                ColdOp::VectorSearch => {
                    let h = rt.block_on(async {
                        let reader = cache.reader(uri).await.expect("cold reader");
                        let vec = reader.vec().expect("vector reader present");
                        vec.search("v", query, TOP_K, nprobe, DEFAULT_RERANK_MULT)
                            .await
                            .expect("cold vector_search")
                    });
                    std::hint::black_box(h);
                }
                ColdOp::Bm25Search => {
                    let h = rt.block_on(async {
                        let reader = cache.reader(uri).await.expect("cold reader");
                        reader
                            .bm25_hits_async(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                            .await
                            .expect("cold bm25_search")
                    });
                    std::hint::black_box(h);
                }
            }
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("{name}[{i}]"), &snap, wall, real_s3);
            let latency = if real_s3 {
                wall
            } else {
                let (raw_blocking, model_blocking, _) =
                    request_blocking_spans(&snap.range_log, model);
                wall.checked_sub(raw_blocking).unwrap_or(Duration::ZERO) + model_blocking
            };
            let cost = cost_model
                .cost_for(snap.head_count, snap.range_count, snap.range_bytes())
                .total_usd();
            latencies.push(latency);
            costs.push(cost);
            // Let the previous iter's background fill stop touching the
            // backend before the next cold timing starts.
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }
        latencies.sort_unstable();
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ColdPhase {
            p50: latencies[latencies.len() / 2],
            cost_usd: costs[costs.len() / 2],
        }
    }

    /// Probe raw s3s-fs / `S3StorageProvider` round-trip latency
    /// for three range sizes (header-sized, MiB-sized, chunk-sized)
    /// then exercise the cold-open + cold-first-search paths with
    /// the counting wrapper installed so we can see exactly what
    /// the cold-fetch coordinator issues against the wire. The
    /// unhinted-vs-hinted A/B + raw-RTT probes here go beyond the
    /// always-on summary `run()` emits; kept as an opt-in deep dive.
    #[allow(dead_code)]
    pub fn diagnose_s3s_fs_cold_path() {
        let rt = Runtime::new().expect("tokio runtime");
        let superfile = superfile_bytes();
        let n = quick_iter_n_docs();
        let nprobe = BENCH_NPROBE;
        let query = query_vector().to_vec();

        let (_addr, _fs_root, raw_storage, uri) = rt.block_on(setup_s3_fixture(superfile));
        let storage = Arc::new(CountingStorage::new(raw_storage));
        let storage_dyn: Arc<dyn StorageProvider> =
            Arc::clone(&storage) as Arc<dyn StorageProvider>;
        let path = uri.storage_path();

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
                total.as_secs_f64() / ITERS as f64 * MS_PER_SEC,
            );
        }
        storage.reset();

        // Build the SubsectionOffsets the manifest would carry,
        // so we can A/B the cold-open path unhinted (2-RTT
        // sequential) vs hinted (1-RTT parallel prefetch).
        let offsets = build_offsets_from_bytes(superfile);
        eprintln!(
            "[diag] manifest hints: total={} B  vec={:?}  fts={:?}",
            offsets.total_size, offsets.vec, offsets.fts
        );

        // ── Phase 2a: cold-open UNHINTED (no manifest hints, 2 RTTs) ───────────
        eprintln!("[diag] === cold-open UNHINTED via cache.reader (3 fresh-cache iters) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let t0 = Instant::now();
            let _reader = rt.block_on(cache.reader(&uri)).expect("cold reader");
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_open_unhinted[{i}]"), &snap, wall, false);
            // Let the previous iter's bg fill stop touching s3s-fs
            // before the next cold timing starts, so contention
            // doesn't poison the measurement. The `sleep` itself
            // must be `await`ed inside `block_on` so it enters
            // the runtime context.
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 2b: cold-open HINTED (1 RTT parallel) ─────────────
        eprintln!(
            "[diag] === cold-open HINTED via cache.reader_with_hints (3 fresh-cache iters) ==="
        );
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let off_ref = offsets.clone();
            let t0 = Instant::now();
            let _reader = rt
                .block_on(cache.reader_with_hints(&uri, Some(&off_ref)))
                .expect("cold reader");
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_open_hinted[{i}]"), &snap, wall, false);
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 3a: cold first search UNHINTED ────────────────────
        eprintln!("[diag] === cold first search UNHINTED (nprobe={nprobe}, top={TOP_K}) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let q = query.clone();
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache.reader(&uri).await.expect("cold reader");
                let vec = reader.vec().expect("vector reader present");
                vec.search("v", &q, TOP_K, nprobe, DEFAULT_RERANK_MULT)
                    .await
                    .expect("cold vector_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(
                &format!("cold_first_search_unhinted[{i}]"),
                &snap,
                wall,
                false,
            );
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 3b: cold first search HINTED ──────────────────────
        eprintln!("[diag] === cold first search HINTED (nprobe={nprobe}, top={TOP_K}) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let q = query.clone();
            let off_ref = offsets.clone();
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache
                    .reader_with_hints(&uri, Some(&off_ref))
                    .await
                    .expect("cold reader");
                let vec = reader.vec().expect("vector reader present");
                vec.search("v", &q, TOP_K, nprobe, DEFAULT_RERANK_MULT)
                    .await
                    .expect("cold vector_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(
                &format!("cold_first_search_hinted[{i}]"),
                &snap,
                wall,
                false,
            );
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 4a: cold first BM25 UNHINTED ──────────────────────
        eprintln!("[diag] === cold first BM25 UNHINTED (term={FTS_QUERY_TERM}, top={TOP_K}) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache.reader(&uri).await.expect("cold reader");
                reader
                    .bm25_hits_async(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                    .await
                    .expect("cold bm25_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(
                &format!("cold_first_bm25_unhinted[{i}]"),
                &snap,
                wall,
                false,
            );
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 4b: cold first BM25 HINTED ────────────────────────
        eprintln!("[diag] === cold first BM25 HINTED (term={FTS_QUERY_TERM}, top={TOP_K}) ===");
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let off_ref = offsets.clone();
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache
                    .reader_with_hints(&uri, Some(&off_ref))
                    .await
                    .expect("cold reader");
                reader
                    .bm25_hits_async(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                    .await
                    .expect("cold bm25_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(&format!("cold_first_bm25_hinted[{i}]"), &snap, wall, false);
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        // ── Phase 4c: cold first multi-term BM25 HINTED ─────────────
        eprintln!(
            "[diag] === cold first BM25 HINTED multi-term (query=\"{FTS_MULTI_QUERY}\", top={TOP_K}) ==="
        );
        for i in 0..3 {
            let before = storage.snapshot();
            let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
            let off_ref = offsets.clone();
            let t0 = Instant::now();
            let _hits = rt.block_on(async {
                let reader = cache
                    .reader_with_hints(&uri, Some(&off_ref))
                    .await
                    .expect("cold reader");
                reader
                    .bm25_hits_async(FTS_COLUMN, FTS_MULTI_QUERY, TOP_K, BoolMode::Or)
                    .await
                    .expect("cold multi-term bm25_search")
            });
            let wall = t0.elapsed();
            let snap = storage.snapshot().diff(&before);
            report(
                &format!("cold_first_bm25_hinted_multi[{i}]"),
                &snap,
                wall,
                false,
            );
            rt.block_on(async { tokio::time::sleep(Duration::from_millis(200)).await });
            drop(cache);
            drop(cache_dir);
        }

        eprintln!(
            "[diag] === scale: n_docs={n}, superfile_size={} MiB ===",
            superfile.len() / BYTES_PER_MIB
        );
    }

    /// Same cold-path diagnostic as `diagnose_s3s_fs_cold_path`,
    /// but against actual AWS S3 using the normal `S3StorageProvider`.
    ///
    /// Invocation:
    ///
    /// ```text
    /// INFINO_DIAG_REAL_S3=1 \
    /// INFINO_REAL_S3_BUCKET=my-bucket \
    /// AWS_REGION=us-east-1 \
    /// cargo bench -- object-store
    /// ```
    pub fn diagnose_real_s3_cold_path() {
        let rt = Runtime::new().expect("tokio runtime");
        let superfile = superfile_bytes();
        let n = quick_iter_n_docs();
        let nprobe = BENCH_NPROBE;
        let query = query_vector().to_vec();
        let bucket = std::env::var("INFINO_REAL_S3_BUCKET")
            .or_else(|_| std::env::var("INFINO_TEST_REAL_S3_BUCKET"))
            .expect("set INFINO_REAL_S3_BUCKET or INFINO_TEST_REAL_S3_BUCKET");
        let prefix_root = std::env::var("INFINO_REAL_S3_PREFIX")
            .unwrap_or_else(|_| "infino-real-s3-bench".to_string());
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        );
        let prefix = format!("{}/{}", prefix_root.trim_matches('/'), unique);
        let uri = SuperfileUri::new_v4();
        let path = uri.storage_path();

        eprintln!(
            "[diag-real-s3] bucket={bucket} prefix={prefix} path={path} n_docs={n} size={} MiB",
            superfile.len() / BYTES_PER_MIB
        );

        let raw_storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_prefix(&bucket, &prefix).expect("real S3 provider"),
        );
        rt.block_on(raw_storage.put_atomic(&path, superfile.clone()))
            .expect("upload superfile to real S3");

        let storage = Arc::new(CountingStorage::new(Arc::clone(&raw_storage)));
        let storage_dyn: Arc<dyn StorageProvider> =
            Arc::clone(&storage) as Arc<dyn StorageProvider>;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eprintln!("[diag-real-s3] === raw AWS S3 range RTT probes ===");
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
                        .expect("real S3 raw range");
                    total += t0.elapsed();
                }
                eprintln!(
                    "[diag-real-s3] raw_get_range[{label:<14}] len={:>8} B  avg={:>6.2} ms over {ITERS} iters",
                    len,
                    total.as_secs_f64() / ITERS as f64 * MS_PER_SEC,
                );
            }
            storage.reset();

            let offsets = build_offsets_from_bytes(superfile);
            eprintln!(
                "[diag-real-s3] manifest hints: total={} B  vec={:?}  fts={:?}",
                offsets.total_size, offsets.vec, offsets.fts
            );

            eprintln!("[diag-real-s3] === cold-open HINTED via real S3 (3 fresh-cache iters) ===");
            for i in 0..3 {
                let before = storage.snapshot();
                let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
                let off_ref = offsets.clone();
                let t0 = Instant::now();
                let _reader = rt
                    .block_on(cache.reader_with_hints(&uri, Some(&off_ref)))
                    .expect("real S3 cold reader");
                let wall = t0.elapsed();
                let snap = storage.snapshot().diff(&before);
                report(&format!("real_s3_cold_open_hinted[{i}]"), &snap, wall, true);
                rt.block_on(async { tokio::time::sleep(Duration::from_millis(500)).await });
                drop(cache);
                drop(cache_dir);
            }

            eprintln!(
                "[diag-real-s3] === cold first vector HINTED (nprobe={nprobe}, top={TOP_K}) ==="
            );
            for i in 0..3 {
                let before = storage.snapshot();
                let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
                let q = query.clone();
                let off_ref = offsets.clone();
                let t0 = Instant::now();
                let _hits = rt.block_on(async {
                    let reader = cache
                        .reader_with_hints(&uri, Some(&off_ref))
                        .await
                        .expect("real S3 cold reader");
                    let vec = reader.vec().expect("vector reader present");
                    vec.search("v", &q, TOP_K, nprobe, DEFAULT_RERANK_MULT)
                        .await
                        .expect("real S3 cold vector_search")
                });
                let wall = t0.elapsed();
                let snap = storage.snapshot().diff(&before);
                report(
                    &format!("real_s3_cold_first_search_hinted[{i}]"),
                    &snap,
                    wall,
                    true,
                );
                rt.block_on(async { tokio::time::sleep(Duration::from_millis(500)).await });
                drop(cache);
                drop(cache_dir);
            }

            eprintln!(
                "[diag-real-s3] === cold first BM25 HINTED (term={FTS_QUERY_TERM}, top={TOP_K}) ==="
            );
            for i in 0..3 {
                let before = storage.snapshot();
                let (cache_dir, cache) = fresh_cache(Arc::clone(&storage_dyn));
                let off_ref = offsets.clone();
                let t0 = Instant::now();
                let _hits = rt.block_on(async {
                    let reader = cache
                        .reader_with_hints(&uri, Some(&off_ref))
                        .await
                        .expect("real S3 cold reader");
                    reader
                        .bm25_hits_async(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                        .await
                        .expect("real S3 cold bm25_search")
                });
                let wall = t0.elapsed();
                let snap = storage.snapshot().diff(&before);
                report(
                    &format!("real_s3_cold_first_bm25_hinted[{i}]"),
                    &snap,
                    wall,
                    true,
                );
                rt.block_on(async { tokio::time::sleep(Duration::from_millis(500)).await });
                drop(cache);
                drop(cache_dir);
            }

            eprintln!(
                "[diag-real-s3] === scale: n_docs={n}, superfile_size={} MiB ===",
                superfile.len() / (1024 * 1024)
            );
        }));

        let cleanup = rt.block_on(raw_storage.delete(&path));
        eprintln!("[diag-real-s3] cleanup path={path} result={cleanup:?}");
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    /// Production-shape real S3 diagnostic: build a unified
    /// vector+FTS supertable through `SupertableOptions::apply_config`,
    /// commit to S3, then reopen from a fresh config-backed handle and
    /// time cold open, cold vector, cold BM25, and warm repeated reads.
    pub fn diagnose_real_s3_supertable_e2e() {
        let rt = Runtime::new().expect("tokio runtime");
        let n = quick_iter_n_docs();
        let nprobe = BENCH_NPROBE;
        let bucket = std::env::var("INFINO_REAL_S3_BUCKET")
            .or_else(|_| std::env::var("INFINO_TEST_REAL_S3_BUCKET"))
            .expect("set INFINO_REAL_S3_BUCKET or INFINO_TEST_REAL_S3_BUCKET");
        let prefix_root = std::env::var("INFINO_REAL_S3_PREFIX")
            .unwrap_or_else(|_| "infino-real-s3-bench".to_string());
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        );
        let prefix = format!("{}/{}", prefix_root.trim_matches('/'), unique);
        let cache_dir = TempDir::new().expect("real S3 supertable cache dir");
        let cfg = real_s3_supertable_config(&bucket, &prefix, cache_dir.path());
        eprintln!(
            "[diag-real-s3-supertable] bucket={bucket} prefix={prefix} n_docs={n} dim={}",
            crate::corpus::DIM
        );

        let cleanup_keys = Arc::new(Mutex::new(Vec::new()));
        let cleanup_keys_for_run = Arc::clone(&cleanup_keys);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.block_on(async move {
            let build_t0 = Instant::now();
            {
                let producer = Supertable::create(
                    real_s3_supertable_options()
                        .apply_config(&cfg)
                        .expect("apply real S3 config to producer"),
                )
                .expect("create real S3 producer");
                let mut writer = producer.writer().expect("real S3 producer writer");
                append_unified_supertable_batches(&mut writer, n);
                writer.commit().expect("commit unified supertable to real S3");
                eprintln!(
                    "[diag-real-s3-supertable] producer commit OK; manifest_id={} build_and_commit_ms={:.1}",
                    producer.manifest_id(),
                    build_t0.elapsed().as_secs_f64() * MS_PER_SEC
                );
            }

            let open_t0 = Instant::now();
            let consumer = Supertable::open(
                real_s3_supertable_options()
                    .apply_config(&cfg)
                    .expect("apply real S3 config to consumer")
                    .with_read_consistency(infino::supertable::options::Consistency::Snapshot),
            )
            .expect("open unified supertable from real S3");
            let cold_open = open_t0.elapsed();
            let reader = consumer.reader();
            eprintln!(
                "[diag-real-s3-supertable] cold_open wall={:.1} ms manifest_id={} n_superfiles={} n_docs_total={}",
                cold_open.as_secs_f64() * MS_PER_SEC,
                consumer.manifest_id(),
                reader.n_superfiles(),
                reader.n_docs_total()
            );

            {
                let manifest = reader.manifest();
                let mut keys = cleanup_keys_for_run.lock().unwrap();
                keys.push("_supertable/current".to_string());
                keys.push(infino::supertable::manifest::commit::list_uri(
                    consumer.manifest_id(),
                ));
                if let Some(list) = &manifest.list {
                    keys.extend(list.parts.iter().map(|p| p.uri.clone()));
                }
                keys.extend(
                    manifest
                        .superfiles
                        .iter()
                        .map(|entry| entry.uri.storage_path()),
                );
            }

            let query = query_vector().to_vec();
            let vec_t0 = Instant::now();
            let vec_hits = consumer
                .reader()
                .vector_search(
                    VEC_COLUMN,
                    &query,
                    TOP_K,
                    VectorSearchOptions::new().with_nprobe(nprobe),
                    None,
                )
                .expect("cold vector search over real S3 supertable");
            let cold_vec = vec_t0.elapsed();
            eprintln!(
                "[diag-real-s3-supertable] cold_vector wall={:.1} ms hits={} nprobe={nprobe}",
                cold_vec.as_secs_f64() * MS_PER_SEC,
                vec_hits.len()
            );

            let bm25_t0 = Instant::now();
            let bm25_hits = consumer
                .reader()
                .bm25_hits(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                .expect("cold BM25 over real S3 supertable");
            let cold_bm25 = bm25_t0.elapsed();
            eprintln!(
                "[diag-real-s3-supertable] cold_bm25 wall={:.1} ms hits={} query={FTS_QUERY_TERM}",
                cold_bm25.as_secs_f64() * MS_PER_SEC,
                bm25_hits.len()
            );

            tokio::time::sleep(Duration::from_secs(2)).await;

            let warm_vec_t0 = Instant::now();
            let warm_vec_hits = consumer
                .reader()
                .vector_search(
                    VEC_COLUMN,
                    &query,
                    TOP_K,
                    VectorSearchOptions::new().with_nprobe(nprobe),
                    None,
                )
                .expect("warm vector search over real S3 supertable");
            let warm_vec = warm_vec_t0.elapsed();
            let warm_bm25_t0 = Instant::now();
            let warm_bm25_hits = consumer
                .reader()
                .bm25_hits(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                .expect("warm BM25 over real S3 supertable");
            let warm_bm25 = warm_bm25_t0.elapsed();
            let cache_stats = consumer
                .options()
                .disk_cache
                .as_ref()
                .expect("real S3 config should attach disk cache")
                .stats();
            eprintln!(
                "[diag-real-s3-supertable] warm_vector wall={:.1} ms hits={} | warm_bm25 wall={:.1} ms hits={} | cache_stats={cache_stats:?}",
                warm_vec.as_secs_f64() * MS_PER_SEC,
                warm_vec_hits.len(),
                warm_bm25.as_secs_f64() * MS_PER_SEC,
                warm_bm25_hits.len()
            );
            })
        }));

        let cleanup_storage =
            S3StorageProvider::new_with_prefix(&bucket, &prefix).expect("real S3 cleanup provider");
        let keys = cleanup_keys.lock().unwrap().clone();
        let cleanup_result = rt.block_on(async {
            for key in &keys {
                let _ = cleanup_storage.delete(key).await;
            }
            cleanup_storage.delete("_supertable/current").await
        });
        eprintln!("[diag-real-s3-supertable] cleanup prefix={prefix} result={cleanup_result:?}");
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    fn real_s3_supertable_config(
        bucket: &str,
        prefix: &str,
        cache_root: &std::path::Path,
    ) -> Config {
        Config {
            supertable: SupertableSettings::default(),
            storage: StorageSettings {
                backend: StorageBackend::S3,
                bucket: Some(bucket.to_string()),
                prefix: prefix.to_string(),
                disk_cache_root: Some(cache_root.to_path_buf()),
                disk_budget_bytes: 8 << 30,
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

    fn real_s3_supertable_options() -> SupertableOptions {
        let schema = Arc::new(Schema::new(vec![
            Field::new(FTS_COLUMN, DataType::LargeUtf8, false),
            Field::new(
                VEC_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    crate::corpus::DIM as i32,
                ),
                false,
            ),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig {
                column: FTS_COLUMN.into(),
            }],
            vec![VectorConfig {
                column: VEC_COLUMN.into(),
                dim: crate::corpus::DIM,
                n_cent: crate::corpus::n_cent(quick_iter_n_docs()),
                rot_seed: ROT_SEED,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8ResidualEpsilon,
            }],
            Some(default_tokenizer()),
        )
        .expect("real S3 supertable options")
    }

    fn append_unified_supertable_batches(
        writer: &mut infino::supertable::writer::SupertableWriter,
        n: usize,
    ) {
        let n_cent = crate::corpus::n_cent(n);
        let dim = crate::corpus::DIM;
        let vectors_mmap = crate::corpus::MmapVectorCorpus::generate(n, n_cent, CORPUS_SEED, true);
        let vectors = vectors_mmap.as_slice();
        let text = crate::corpus::MmapTextCorpus::generate(n, 1);
        let schema = Arc::new(Schema::new(vec![
            Field::new(FTS_COLUMN, DataType::LargeUtf8, false),
            Field::new(
                VEC_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        const CHUNK: usize = 65_536;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        for start in (0..n).step_by(CHUNK) {
            let len = CHUNK.min(n - start);
            let titles = LargeStringArray::from(
                (start..start + len)
                    .map(|i| text.doc(i))
                    .collect::<Vec<_>>(),
            );
            let values = Float32Array::from(vectors[start * dim..(start + len) * dim].to_vec());
            let vectors = FixedSizeListArray::try_new(
                Arc::clone(&item_field),
                dim as i32,
                Arc::new(values) as Arc<dyn Array>,
                None,
            )
            .expect("vector fixed-size-list array");
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(titles), Arc::new(vectors)])
                    .expect("unified supertable batch");
            writer
                .append(&batch)
                .expect("append unified supertable batch");
        }
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
        let total_size = bytes.len() as u64;
        let vec_open_ranges = vec
            .and_then(|(off, len)| vector_open_ranges(bytes, off, len))
            .unwrap_or_default();
        let fts_open_ranges = fts
            .and_then(|(off, len)| fts_open_ranges(bytes, off, len))
            .unwrap_or_default();

        // Mirror the writer's open-blob capture: parquet tail
        // (64 KiB) + each vec/fts open range, sliced inline so the
        // diagnostic exercises the zero-open-GET cold path.
        const PARQUET_TAIL_SPEC: u64 = 64 * 1024;
        let mut open_blob: Vec<(u64, Vec<u8>)> = Vec::new();
        let parquet_tail_len = PARQUET_TAIL_SPEC.min(total_size);
        let parquet_tail_start = total_size.saturating_sub(parquet_tail_len);
        let slice = |off: u64, len: u64| -> Option<Vec<u8>> {
            let start = off as usize;
            let end = start.checked_add(len as usize)?;
            bytes.get(start..end).map(|s| s.to_vec())
        };
        let mut ok = true;
        if parquet_tail_len > 0 {
            match slice(parquet_tail_start, parquet_tail_len) {
                Some(b) => open_blob.push((parquet_tail_start, b)),
                None => ok = false,
            }
        }
        if ok {
            for &(off, len) in vec_open_ranges.iter().chain(fts_open_ranges.iter()) {
                match slice(off, len) {
                    Some(b) => open_blob.push((off, b)),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            open_blob.clear();
        }

        SubsectionOffsets {
            total_size,
            vec,
            fts,
            vec_open_ranges,
            fts_open_ranges,
            open_blob,
        }
    }

    fn vector_open_ranges(bytes: &[u8], off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
        const OUTER_HEADER_SIZE: usize = 32;
        const DIR_ENTRY_SIZE: usize = 64;
        const SUB_HEADER_SIZE: usize = 56;
        let start = off as usize;
        let end = start.checked_add(len as usize)?;
        let blob = bytes.get(start..end)?;
        if blob.len() < OUTER_HEADER_SIZE + 4 {
            return None;
        }
        let n_columns = read_u32_le(blob.get(12..16)?) as usize;
        let dir_offset = read_u64_le(blob.get(24..32)?) as usize;
        let dir_size = n_columns.checked_mul(DIR_ENTRY_SIZE)?;
        let dir_end = dir_offset.checked_add(dir_size)?.checked_add(4)?;
        let dir = blob.get(dir_offset..dir_offset + dir_size)?;

        let mut ranges = vec![(off, OUTER_HEADER_SIZE as u64)];
        ranges.push((off + dir_offset as u64, (dir_size + 4) as u64));
        for i in 0..n_columns {
            let entry = i * DIR_ENTRY_SIZE;
            let subsection_off = read_u64_le(dir.get(entry + 24..entry + 32)?) as usize;
            let subsection_len = read_u64_le(dir.get(entry + 32..entry + 40)?) as usize;
            let codec_meta_off = read_u32_le(dir.get(entry + 56..entry + 60)?) as usize;
            let codec_meta_size = read_u32_le(dir.get(entry + 60..entry + 64)?) as usize;
            if subsection_off.checked_add(SUB_HEADER_SIZE)? > blob.len()
                || subsection_off.checked_add(subsection_len)? > blob.len()
            {
                return None;
            }
            ranges.push((off + subsection_off as u64, SUB_HEADER_SIZE as u64));
            let sub = blob.get(subsection_off..subsection_off + subsection_len)?;
            let centroids_off = read_u64_le(sub.get(32..40)?) as usize;
            let cluster_idx_off = read_u64_le(sub.get(40..48)?) as usize;
            let n_cent = read_u32_le(dir.get(entry + 8..entry + 12)?) as usize;
            let cluster_idx_end = cluster_idx_off.checked_add(n_cent * 8)?;
            if centroids_off < SUB_HEADER_SIZE || cluster_idx_end > subsection_len {
                return None;
            }
            ranges.push((
                off + subsection_off as u64 + centroids_off as u64,
                (cluster_idx_end - centroids_off) as u64,
            ));
            if codec_meta_size > 0 {
                let meta_end = codec_meta_off.checked_add(codec_meta_size)?;
                if meta_end > subsection_len {
                    return None;
                }
            }
        }
        if dir_end > blob.len() {
            return None;
        }
        Some(merge_ranges(ranges))
    }

    fn fts_open_ranges(bytes: &[u8], off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
        const FTS_HEADER_SIZE: usize = 48;
        let start = off as usize;
        let end = start.checked_add(len as usize)?;
        let blob = bytes.get(start..end)?;
        if blob.len() < FTS_HEADER_SIZE {
            return None;
        }
        let postings_offset = read_u64_le(blob.get(32..40)?) as usize;
        let doc_lengths_offset = read_u64_le(blob.get(40..48)?) as usize;
        if postings_offset > blob.len()
            || doc_lengths_offset > blob.len()
            || postings_offset > doc_lengths_offset
        {
            return None;
        }
        Some(merge_ranges(vec![
            (off, postings_offset as u64),
            (
                off + doc_lengths_offset as u64,
                (blob.len() - doc_lengths_offset) as u64,
            ),
        ]))
    }

    fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
        ranges.retain(|&(_, len)| len > 0);
        ranges.sort_unstable_by_key(|&(off, _)| off);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        for (off, len) in ranges {
            let end = off + len;
            if let Some((last_off, last_len)) = merged.last_mut() {
                let last_end = *last_off + *last_len;
                if off <= last_end {
                    *last_len = (*last_len).max(end - *last_off);
                    continue;
                }
            }
            merged.push((off, len));
        }
        merged
    }

    fn read_u32_le(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes.try_into().expect("u32 slice length"))
    }

    fn read_u64_le(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes.try_into().expect("u64 slice length"))
    }

    /// Times warm `reader.bm25_search` / `reader.vector_search`
    /// (kernel-direct) vs `consumer.reader().query_sql("SELECT _id FROM
    /// bm25_search(...)")` / `query_sql("... vector_search ...")`
    /// (DataFusion path) side-by-side on the same warm Supertable
    /// over an in-process `s3s-fs`. Prints min / p50 / p95 / mean
    /// for each path plus the p50 dispatch-tax delta to stderr.
    ///
    /// Storage backend doesn't matter for this measurement —
    /// after warm-up both paths read from mmap (zero S3 GETs).
    /// The delta is the per-call cost of `SessionContext::new()` +
    /// TVF re-registration + SQL parse/plan + RecordBatch glue
    /// that `query_sql` pays on top of the kernel.
    ///
    /// Knobs:
    ///   `INFINO_DIAG_QUERY_SQL_ITERS` (default 50) — iters per path.
    ///   `INFINO_BENCH_FULL=1` — corpus = 1M (else 100K).
    pub fn diagnose_query_sql_overhead() {
        use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, LruPolicy};
        use std::collections::HashSet;

        let rt = Runtime::new().expect("tokio runtime");
        let n = quick_iter_n_docs();
        let iters: usize = std::env::var("INFINO_DIAG_QUERY_SQL_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        eprintln!(
            "[diag-qsql-overhead] n_docs={n} iters={iters} \
             (override via INFINO_DIAG_QUERY_SQL_ITERS, INFINO_BENCH_FULL=1 for 1M)"
        );

        // 1. Spawn s3s-fs + storage provider.
        let (addr, _fs_root) = rt.block_on(spawn_s3s_fs());
        let endpoint = format!("http://{addr}");
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider"),
        );

        // 2. Disk cache (so warm == mmap, not re-fetch).
        let cache_dir = TempDir::new().expect("cache tempdir");
        let cache_cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            disk_budget_bytes: 4u64 << 30,
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
        let cache = DiskCacheStore::new(Arc::clone(&storage), cache_cfg, pinned).expect("cache");

        // 3. Producer: write n docs through Supertable's writer.
        eprintln!("[diag-qsql-overhead] writing {n}-doc Supertable to s3s-fs ...");
        let build_t0 = Instant::now();
        rt.block_on(async {
            let producer =
                Supertable::create(real_s3_supertable_options().with_storage(Arc::clone(&storage)))
                    .expect("create producer Supertable");
            let mut writer = producer.writer().expect("producer writer");
            append_unified_supertable_batches(&mut writer, n);
            writer.commit().expect("commit Supertable to s3s-fs");
        });
        eprintln!(
            "[diag-qsql-overhead] commit OK in {:.1} s",
            build_t0.elapsed().as_secs_f64()
        );

        // 4. Consumer with disk cache attached.
        let consumer = rt.block_on(async {
            Supertable::open(
                real_s3_supertable_options()
                    .with_storage(Arc::clone(&storage))
                    .with_disk_cache(Arc::clone(&cache)),
            )
            .expect("Supertable::open from s3s-fs")
        });
        // 5. Warm the cache: cold pass + mmap-promotion sleep.
        eprintln!("[diag-qsql-overhead] warming cache (cold pass + 2s mmap promotion sleep)");
        let q = query_vector().to_vec();
        let _ = consumer
            .reader()
            .bm25_hits(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
            .expect("warm-up bm25");
        let _ = consumer
            .reader()
            .vector_search(
                VEC_COLUMN,
                &q,
                TOP_K,
                VectorSearchOptions::new().with_nprobe(BENCH_NPROBE),
                None,
            )
            .expect("warm-up vector");
        rt.block_on(async { tokio::time::sleep(Duration::from_secs(2)).await });

        // 6. Pre-warm the query_sql path: lazy-allocates the
        //    Supertable's internal sql_runtime and warms any
        //    DataFusion lazy state so the first timed iter
        //    isn't contaminated by one-time setup cost.
        let q_csv: String = q
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let bm25_sql =
            format!("SELECT _id FROM bm25_search('{FTS_COLUMN}', '{FTS_QUERY_TERM}', {TOP_K})");
        let vec_sql = format!("SELECT _id FROM vector_search('{VEC_COLUMN}', '{q_csv}', {TOP_K})");
        // Score-only projection skips `resolve_hits` -> `resolve_columns`
        // (no scalar decode, no per-superfile superfile_reader open).
        // Difference vs `_id` projection isolates resolve_hits cost.
        let bm25_score_sql =
            format!("SELECT score FROM bm25_search('{FTS_COLUMN}', '{FTS_QUERY_TERM}', {TOP_K})");
        let vec_score_sql =
            format!("SELECT score FROM vector_search('{VEC_COLUMN}', '{q_csv}', {TOP_K})");
        let _ = consumer
            .reader()
            .query_sql(&bm25_sql)
            .expect("warm-up query_sql bm25");
        let _ = consumer
            .reader()
            .query_sql(&vec_sql)
            .expect("warm-up query_sql vector");

        // 7. Time both paths.
        let opts = VectorSearchOptions::new().with_nprobe(BENCH_NPROBE);

        let mut kernel_bm25: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .bm25_hits(FTS_COLUMN, FTS_QUERY_TERM, TOP_K, BoolMode::Or)
                .expect("kernel bm25");
            kernel_bm25.push(t.elapsed());
        }
        let mut qsql_bm25: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .query_sql(&bm25_sql)
                .expect("query_sql bm25");
            qsql_bm25.push(t.elapsed());
        }
        let mut kernel_vec: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .vector_search(VEC_COLUMN, &q, TOP_K, opts, None)
                .expect("kernel vector");
            kernel_vec.push(t.elapsed());
        }
        let mut qsql_vec: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .query_sql(&vec_sql)
                .expect("query_sql vector");
            qsql_vec.push(t.elapsed());
        }

        // Decompose query_sql into parse+plan (ctx.sql) vs
        // execute (DataFrame::collect) so we can see where the
        // remaining dispatch time goes after the SessionContext
        // cache hit. Goes through the same cached SessionContext
        // the public query_sql uses — no rebuild per iter.
        let cached_ctx = consumer.__debug_cached_session();
        let mut bm25_parse_plan: Vec<Duration> = Vec::with_capacity(iters);
        let mut bm25_execute: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let df = rt
                .block_on(cached_ctx.sql(&bm25_sql))
                .expect("ctx.sql bm25");
            bm25_parse_plan.push(t0.elapsed());
            let t1 = Instant::now();
            let _ = rt.block_on(df.collect()).expect("collect bm25");
            bm25_execute.push(t1.elapsed());
        }
        let mut vec_parse_plan: Vec<Duration> = Vec::with_capacity(iters);
        let mut vec_execute: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let df = rt
                .block_on(cached_ctx.sql(&vec_sql))
                .expect("ctx.sql vector");
            vec_parse_plan.push(t0.elapsed());
            let t1 = Instant::now();
            let _ = rt.block_on(df.collect()).expect("collect vector");
            vec_execute.push(t1.elapsed());
        }
        let mut bm25_score_total: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .query_sql(&bm25_score_sql)
                .expect("query_sql bm25 score");
            bm25_score_total.push(t.elapsed());
        }
        let mut vec_score_total: Vec<Duration> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = consumer
                .reader()
                .query_sql(&vec_score_sql)
                .expect("query_sql vector score");
            vec_score_total.push(t.elapsed());
        }

        // 8. Stats + report.
        fn stats(samples: &mut [Duration]) -> (Duration, Duration, Duration, Duration) {
            samples.sort();
            let n = samples.len();
            let sum_ns: u128 = samples.iter().map(|d| d.as_nanos()).sum();
            let mean = Duration::from_nanos((sum_ns / n as u128) as u64);
            (
                samples[0],
                samples[n / 2],
                samples[(n * 95 / 100).min(n - 1)],
                mean,
            )
        }
        let fmt = |d: Duration| -> String {
            let us = d.as_secs_f64() * 1e6;
            if us < 1000.0 {
                format!("{us:>9.1} µs")
            } else {
                format!("{:>9.2} ms", us / 1000.0)
            }
        };

        let (kb_min, kb_p50, kb_p95, kb_mean) = stats(&mut kernel_bm25);
        let (qb_min, qb_p50, qb_p95, qb_mean) = stats(&mut qsql_bm25);
        let (kv_min, kv_p50, kv_p95, kv_mean) = stats(&mut kernel_vec);
        let (qv_min, qv_p50, qv_p95, qv_mean) = stats(&mut qsql_vec);

        eprintln!();
        eprintln!(
            "[diag-qsql-overhead] === kernel vs query_sql (warm, n={n} docs, iters={iters}) ==="
        );
        eprintln!(
            "[diag-qsql-overhead]                          min        p50        p95       mean"
        );
        eprintln!(
            "[diag-qsql-overhead] BM25 kernel        {} {} {} {}",
            fmt(kb_min),
            fmt(kb_p50),
            fmt(kb_p95),
            fmt(kb_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] BM25 query_sql     {} {} {} {}",
            fmt(qb_min),
            fmt(qb_p50),
            fmt(qb_p95),
            fmt(qb_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] BM25 dispatch tax  {} (p50 query_sql − kernel)",
            fmt(qb_p50.saturating_sub(kb_p50)),
        );
        eprintln!();
        eprintln!(
            "[diag-qsql-overhead] vec  kernel        {} {} {} {}",
            fmt(kv_min),
            fmt(kv_p50),
            fmt(kv_p95),
            fmt(kv_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  query_sql     {} {} {} {}",
            fmt(qv_min),
            fmt(qv_p50),
            fmt(qv_p95),
            fmt(qv_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  dispatch tax  {} (p50 query_sql − kernel)",
            fmt(qv_p50.saturating_sub(kv_p50)),
        );

        let (bp_min, bp_p50, bp_p95, bp_mean) = stats(&mut bm25_parse_plan);
        let (be_min, be_p50, be_p95, be_mean) = stats(&mut bm25_execute);
        let (vp_min, vp_p50, vp_p95, vp_mean) = stats(&mut vec_parse_plan);
        let (ve_min, ve_p50, ve_p95, ve_mean) = stats(&mut vec_execute);
        eprintln!();
        eprintln!("[diag-qsql-overhead] === decomposition: ctx.sql() vs DataFrame::collect() ===");
        eprintln!(
            "[diag-qsql-overhead] BM25 parse+plan    {} {} {} {}",
            fmt(bp_min),
            fmt(bp_p50),
            fmt(bp_p95),
            fmt(bp_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] BM25 execute       {} {} {} {}",
            fmt(be_min),
            fmt(be_p50),
            fmt(be_p95),
            fmt(be_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  parse+plan    {} {} {} {}",
            fmt(vp_min),
            fmt(vp_p50),
            fmt(vp_p95),
            fmt(vp_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  execute       {} {} {} {}",
            fmt(ve_min),
            fmt(ve_p50),
            fmt(ve_p95),
            fmt(ve_mean),
        );

        let (bs_min, bs_p50, bs_p95, bs_mean) = stats(&mut bm25_score_total);
        let (vs_min, vs_p50, vs_p95, vs_mean) = stats(&mut vec_score_total);
        eprintln!();
        eprintln!("[diag-qsql-overhead] === score-only projection (skips resolve_hits) ===");
        eprintln!(
            "[diag-qsql-overhead] BM25 score-only    {} {} {} {}",
            fmt(bs_min),
            fmt(bs_p50),
            fmt(bs_p95),
            fmt(bs_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] BM25 resolve_hits  {} (p50 _id − p50 score-only)",
            fmt(qb_p50.saturating_sub(bs_p50)),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  score-only    {} {} {} {}",
            fmt(vs_min),
            fmt(vs_p50),
            fmt(vs_p95),
            fmt(vs_mean),
        );
        eprintln!(
            "[diag-qsql-overhead] vec  resolve_hits  {} (p50 _id − p50 score-only)",
            fmt(qv_p50.saturating_sub(vs_p50)),
        );
    }
}
