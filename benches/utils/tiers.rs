// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared hot / cold storage tier helpers for canonical benches.
//!
//! - **Hot**: `Supertable::open` from object storage + `DiskCacheStore` (local cache hits).
//! - **Cold**: fresh disk cache per iteration → object-store range GETs.
//!
//! Default backing store is in-process `s3s-fs`. Set `INFINO_REAL_S3_BUCKET`
//! (or `INFINO_TEST_REAL_S3_BUCKET`) for AWS S3.

use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use infino::supertable::{SuperfileUri, Supertable, SupertableOptions};
use s3s::auth::SimpleAuth;
use s3s::service::S3ServiceBuilder;
use s3s_fs::FileSystem;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

const S3S_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const S3S_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
const S3S_REGION: &str = "us-east-1";

/// Bytes in one gibibyte, for GiB-denominated cache budgets.
const GIB_BYTES: u64 = 1u64 << 30;
/// Bytes in one mebibyte.
const MIB_BYTES: u64 = 1u64 << 20;
/// Default ingest disk-cache budget (GiB) when no env override is set.
const DEFAULT_INGEST_CACHE_GIB: u64 = 8;
/// Auto-sized search cache adds `index_size / this` headroom (+10%).
const INDEX_CACHE_HEADROOM_DIVISOR: u64 = 10;
/// Disk-cache budget (GiB) for single-superfile tier benches.
const SUPERFILE_CACHE_GIB: u64 = 4;
/// Parallel cold-fetch streams used by the bench disk cache.
const BENCH_COLD_FETCH_STREAMS: usize = 8;
/// Cold-fetch range chunk size (8 MiB) used by the bench disk cache.
const BENCH_COLD_FETCH_CHUNK_BYTES: u64 = 8 * MIB_BYTES;
/// Mmap promotion timers disabled in benches (no idle eviction).
const MMAP_TIMER_DISABLED_SECS: u64 = 0;

const SUPERFILE_S3S_BUCKET: &str = "infino-bench-superfile";

/// Storage tier exercised by a search bench row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Hot,
    Cold,
}

impl Tier {
    pub const ALL: [Tier; 2] = [Tier::Hot, Tier::Cold];

    pub fn label(self) -> &'static str {
        match self {
            Tier::Hot => "hot",
            Tier::Cold => "cold",
        }
    }
}

/// Stable report group name for a tiered search bench family (`superfile_vec`, `supertable_fts`, …).
pub fn search_group_name(family: &str, tier: Tier, storage_label: Option<&str>) -> String {
    match tier {
        Tier::Hot => format!("{family}_hot_search"),
        Tier::Cold => {
            let label = storage_label.expect("cold groups need a storage label");
            format!("{family}_{}_search_{label}", tier.label())
        }
    }
}

/// Selected object-store backend for warm/cold tiers.
pub struct StorageFixture {
    pub storage: Arc<dyn StorageProvider>,
    pub storage_label: &'static str,
    pub real_s3: bool,
    /// Real-S3 prefix to delete when the run finishes (`None` for the
    /// auto-cleaned s3s-fs tempdir backend).
    pub cleanup: Option<S3Cleanup>,
    _keepalive: StorageKeepalive,
}

enum StorageKeepalive {
    S3sFs { _fs_root: TempDir },
    RealS3,
}

/// A real-S3 prefix that a bench run created and must delete on exit so it
/// accrues no storage cost. The supertable build writes many objects
/// (segments, manifests, the pointer) under one unique prefix; cleanup
/// lists every key beneath it and deletes them.
#[derive(Clone)]
pub struct S3Cleanup {
    pub bucket: String,
    pub prefix: String,
}

/// Delete every object under a real-S3 bench prefix. Uses a fresh, *un*-prefixed
/// provider on purpose: `list_with_prefix` takes an absolute key prefix (it does
/// not prepend a provider prefix) and `delete` targets the absolute keys it
/// returns verbatim, so both sides agree on the same keyspace.
pub fn cleanup_real_s3_prefix(cleanup: &S3Cleanup) {
    let provider = match S3StorageProvider::new(&cleanup.bucket) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "[tiers] cleanup: cannot open bucket {} ({e}); prefix {} NOT deleted",
                cleanup.bucket, cleanup.prefix
            );
            return;
        }
    };
    let prefix = cleanup.prefix.clone();
    let result: Result<usize, String> = block_on(async move {
        let keys = provider
            .list_with_prefix(&prefix)
            .await
            .map_err(|e| e.to_string())?;
        let n = keys.len();
        for key in &keys {
            provider.delete(key).await.map_err(|e| e.to_string())?;
        }
        Ok(n)
    });
    match result {
        Ok(n) => eprintln!(
            "[tiers] cleanup real S3 prefix={}: deleted {n} objects",
            cleanup.prefix
        ),
        Err(e) => eprintln!(
            "[tiers] cleanup real S3 prefix={}: FAILED ({e}) — objects may remain",
            cleanup.prefix
        ),
    }
}

/// A single superfile committed to object storage (1M tier benches).
pub struct SuperfileCommitted {
    pub storage: Arc<dyn StorageProvider>,
    pub uri: SuperfileUri,
    /// Object key under the storage provider (same bytes the hot
    /// path built — uploaded verbatim for lazy vector open).
    pub object_path: String,
    pub object_size: u64,
    pub storage_label: &'static str,
    pub real_s3: bool,
    pub cleanup_path: Option<String>,
    _keepalive: StorageKeepalive,
}

impl SuperfileCommitted {
    /// Delete the uploaded object when the fixture points at real S3.
    /// s3s-fs fixtures live under a tempdir and are cleaned up by dropping
    /// `_keepalive`, so they do not need object-level deletion.
    pub fn cleanup(&self) {
        let Some(path) = self.cleanup_path.as_deref() else {
            return;
        };
        let storage = Arc::clone(&self.storage);
        let result = block_on(async move { storage.delete(path).await });
        match result {
            Ok(()) => eprintln!("[tiers] cleanup real S3 superfile path={path}: deleted"),
            Err(e) => eprintln!("[tiers] cleanup real S3 superfile path={path}: {e}"),
        }
    }
}

impl Drop for SuperfileCommitted {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// One runtime for the whole bench process. `spawn_s3s_fs` binds its
/// accept loop to this runtime; creating a fresh `Runtime` per
/// `block_on` call would drop the previous one and kill in-process
/// s3s-fs before warm/cold tiers run.
static TIER_RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn tier_runtime() -> &'static Runtime {
    TIER_RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime for tier benches"))
}

pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tier_runtime().block_on(fut)
}

pub fn real_s3_bucket_env() -> Option<String> {
    std::env::var("INFINO_REAL_S3_BUCKET")
        .or_else(|_| std::env::var("INFINO_TEST_REAL_S3_BUCKET"))
        .ok()
}

pub fn real_s3_prefix_root(default: &str) -> String {
    std::env::var("INFINO_REAL_S3_PREFIX").unwrap_or_else(|_| default.to_string())
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

async fn spawn_s3s_fs(s3s_bucket: &str) -> (SocketAddr, TempDir) {
    let fs_root = TempDir::new().expect("s3s-fs root tempdir");
    std::fs::create_dir_all(fs_root.path().join(s3s_bucket)).expect("create bucket dir");

    let fs_backend = FileSystem::new(fs_root.path()).expect("s3s-fs FileSystem");
    let service = {
        let mut b = S3ServiceBuilder::new(fs_backend);
        b.set_auth(SimpleAuth::from_single(S3S_ACCESS_KEY, S3S_SECRET_KEY));
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

async fn backing_store(s3s_bucket: &str, prefix_default: &str) -> StorageFixture {
    if let Some(bucket) = real_s3_bucket_env() {
        let prefix = unique_bench_prefix(&real_s3_prefix_root(prefix_default));
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_prefix(&bucket, &prefix).expect("real S3 provider"),
        );
        eprintln!("[tiers] real S3: bucket={bucket} prefix={prefix}");
        StorageFixture {
            storage,
            storage_label: "real_s3",
            real_s3: true,
            cleanup: Some(S3Cleanup { bucket, prefix }),
            _keepalive: StorageKeepalive::RealS3,
        }
    } else {
        let (addr, fs_root) = spawn_s3s_fs(s3s_bucket).await;
        let endpoint = format!("http://{addr}");
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                s3s_bucket,
                S3S_ACCESS_KEY,
                S3S_SECRET_KEY,
                S3S_REGION,
            )
            .expect("s3s-fs S3StorageProvider"),
        );
        eprintln!(
            "\n\
             ################################################################################\n\
             ##  WARNING: benchmarking against the s3s-fs emulator, NOT real AWS S3.        ##\n\
             ##  The emulator reproduces request count and byte volume, not network         ##\n\
             ##  latency, so warm/cold timings here are not representative of S3.            ##\n\
             ##  Set INFINO_REAL_S3_BUCKET (+ AWS creds) to benchmark against real S3.       ##\n\
             ################################################################################\n\
             [tiers] s3s-fs endpoint={endpoint}  storage_label=s3s_fs  (NOT real S3)\n"
        );
        StorageFixture {
            storage,
            storage_label: "s3s_fs",
            real_s3: false,
            cleanup: None,
            _keepalive: StorageKeepalive::S3sFs { _fs_root: fs_root },
        }
    }
}

/// Error string for the missing-bucket guard. Kept as a constant so the
/// `run()` pre-flight check and this fixture report the same message.
pub const SUPERTABLE_REQUIRES_REAL_S3: &str = "\
the supertable object-store bench requires real AWS S3. Set INFINO_REAL_S3_BUCKET \
(or INFINO_TEST_REAL_S3_BUCKET) to a writable bucket and provide AWS credentials. \
The s3s-fs emulator is not usable here: it does not implement conditional If-Match \
PUTs, which the supertable's multi-commit OCC requires, so every commit after the \
first would lose the CAS. There is no local stand-in — a local filesystem backend \
would not measure object-store ingest or cold-read behavior, which is the whole \
point of this bench.";

/// Supertable-shaped backing store (multi-segment, multi-commit benches).
///
/// **Real S3 only.** Unlike the single-`put_atomic` superfile cold tier, the
/// supertable build commits many times, so its OCC pointer update rides on the
/// conditional `If-Match` PUT (`put_if_match(Some(etag))`). The in-process
/// `s3s-fs` emulator does not implement conditional `If-Match` PUTs (every
/// commit after the first loses the CAS), and a local filesystem backend would
/// not measure the object-store behavior this bench exists for. So this fixture
/// requires `INFINO_REAL_S3_BUCKET` and panics with [`SUPERTABLE_REQUIRES_REAL_S3`]
/// otherwise. The returned fixture carries an [`S3Cleanup`] so the caller can
/// delete the unique prefix when the run finishes.
pub async fn supertable_storage_fixture() -> StorageFixture {
    let bucket = real_s3_bucket_env().expect(SUPERTABLE_REQUIRES_REAL_S3);
    let prefix = unique_bench_prefix(&real_s3_prefix_root("infino-supertable-bench"));
    let storage: Arc<dyn StorageProvider> =
        Arc::new(S3StorageProvider::new_with_prefix(&bucket, &prefix).expect("real S3 provider"));
    eprintln!("[tiers] real S3: bucket={bucket} prefix={prefix}");
    StorageFixture {
        storage,
        storage_label: "real_s3",
        real_s3: true,
        cleanup: Some(S3Cleanup { bucket, prefix }),
        _keepalive: StorageKeepalive::RealS3,
    }
}

/// Upload one superfile blob for superfile-shaped warm/cold benches (1M).
pub async fn commit_superfile(bytes: &Bytes) -> SuperfileCommitted {
    let fixture = backing_store(SUPERFILE_S3S_BUCKET, "infino-superfile-bench").await;
    let uri = SuperfileUri::new_v4();
    let path = uri.storage_path();
    fixture
        .storage
        .put_atomic(&path, bytes.clone())
        .await
        .expect("upload superfile");
    eprintln!(
        "[tiers] superfile committed: {} path={path} ({} MiB)",
        fixture.storage_label,
        bytes.len() / MIB_BYTES as usize
    );
    SuperfileCommitted {
        storage: fixture.storage,
        uri,
        object_path: path.clone(),
        object_size: bytes.len() as u64,
        storage_label: fixture.storage_label,
        real_s3: fixture.real_s3,
        cleanup_path: if fixture.real_s3 { Some(path) } else { None },
        _keepalive: fixture._keepalive,
    }
}

fn env_gib(name: &str, default_gib: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_gib)
}

fn supertable_search_cache_gib() -> Option<u64> {
    std::env::var("INFINO_SUPERTABLE_SEARCH_CACHE_GIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
}

/// Fresh disk cache for ingest producers (8 GiB budget).
///
/// Ingest attaches this cache only to keep segment bytes out of the
/// unbounded in-memory tier; commit-time cache prepopulation is disabled,
/// so this budget is not meant to hold the searchable working set.
pub fn fresh_disk_cache(storage: Arc<dyn StorageProvider>) -> (TempDir, Arc<DiskCacheStore>) {
    fresh_disk_cache_with_mode(
        storage,
        env_gib(
            "INFINO_SUPERTABLE_INGEST_CACHE_GIB",
            DEFAULT_INGEST_CACHE_GIB,
        ) * GIB_BYTES,
        ColdFetchMode::LazyForegroundWithBackgroundFill,
    )
}

/// Fresh disk cache for supertable search consumers.
///
/// Budget selection (first match wins):
/// 1. `INFINO_SUPERTABLE_SEARCH_CACHE_GIB` env var (explicit override).
/// 2. `index_size_bytes + 10%` when the caller knows the total index
///    size from the manifest — ensures the hot bench is truly hot.
/// 3. `INFINO_SUPERTABLE_INGEST_CACHE_GIB` or 8 GiB fallback.
pub fn fresh_supertable_search_cache(
    storage: Arc<dyn StorageProvider>,
    index_size_bytes: Option<u64>,
) -> (TempDir, Arc<DiskCacheStore>) {
    use std::sync::Once;
    static LOG_ONCE: Once = Once::new();

    let budget_bytes = if let Some(explicit_gib) = supertable_search_cache_gib() {
        let b = explicit_gib * GIB_BYTES;
        LOG_ONCE.call_once(|| {
            eprintln!("[tiers] search cache budget = {explicit_gib} GiB (INFINO_SUPERTABLE_SEARCH_CACHE_GIB)");
        });
        b
    } else if let Some(idx) = index_size_bytes.filter(|&s| s > 0) {
        let b = idx + idx / INDEX_CACHE_HEADROOM_DIVISOR;
        LOG_ONCE.call_once(|| {
            eprintln!(
                "[tiers] search cache budget = {:.2} GiB (auto-sized from {:.2} GiB index + 10% headroom)",
                b as f64 / GIB_BYTES as f64,
                idx as f64 / GIB_BYTES as f64,
            );
        });
        b
    } else {
        let gib = env_gib(
            "INFINO_SUPERTABLE_INGEST_CACHE_GIB",
            DEFAULT_INGEST_CACHE_GIB,
        );
        LOG_ONCE.call_once(|| {
            eprintln!("[tiers] search cache budget = {gib} GiB (default)");
        });
        gib * GIB_BYTES
    };
    fresh_disk_cache_with_mode(
        storage,
        budget_bytes,
        ColdFetchMode::LazyForegroundWithBackgroundFill,
    )
}

/// Fresh disk cache for single-superfile tier benches (4 GiB budget).
pub fn fresh_superfile_cache(storage: Arc<dyn StorageProvider>) -> (TempDir, Arc<DiskCacheStore>) {
    fresh_disk_cache_with_mode(
        storage,
        SUPERFILE_CACHE_GIB * GIB_BYTES,
        ColdFetchMode::LazyForegroundWithBackgroundFill,
    )
}

fn fresh_disk_cache_with_mode(
    storage: Arc<dyn StorageProvider>,
    disk_budget_bytes: u64,
    cold_fetch_mode: ColdFetchMode,
) -> (TempDir, Arc<DiskCacheStore>) {
    let dir = TempDir::new().expect("disk cache tempdir");
    let cfg = DiskCacheConfig {
        cache_root: dir.path().to_path_buf(),
        disk_budget_bytes,
        cold_fetch_mode,
        cold_fetch_streams: BENCH_COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: BENCH_COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: MMAP_TIMER_DISABLED_SECS,
        mmap_sweep_interval_secs: MMAP_TIMER_DISABLED_SECS,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: false,
        ..Default::default()
    };
    let cache = DiskCacheStore::new_unpinned(storage, cfg).expect("DiskCacheStore");
    (dir, cache)
}

pub fn consumer_options(
    base: SupertableOptions,
    storage: Arc<dyn StorageProvider>,
    cache: Arc<DiskCacheStore>,
) -> SupertableOptions {
    // Search benches query a static, already-ingested supertable with no
    // concurrent writers. Snapshot consistency keeps the read path free of
    // pointer-GET refreshes so the measured latency is pure query cost; the
    // one-time cold-open manifest read is timed separately.
    base.with_storage(storage)
        .with_disk_cache(cache)
        .with_read_consistency(infino::supertable::options::Consistency::Snapshot)
}

pub fn open_consumer(opts: SupertableOptions) -> Supertable {
    Supertable::open(opts).expect("Supertable::open from object store")
}
