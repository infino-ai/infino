//! Shared hot / cold storage tier helpers for canonical benches.
//!
//! - **Hot**: `Supertable::open` from object storage + `DiskCacheStore` (local cache hits).
//! - **Cold**: fresh disk cache per iteration → object-store range GETs.
//!
//! Default backing store is in-process `s3s-fs`. Set `INFINO_REAL_S3_BUCKET`
//! (or `INFINO_TEST_REAL_S3_BUCKET`) for AWS S3.

use std::collections::HashSet;
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

const SUPERTABLE_S3S_BUCKET: &str = "infino-bench-supertable";
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

/// Criterion group name for a tiered search bench family (`superfile_vec`, `supertable_fts`, …).
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
    _keepalive: StorageKeepalive,
}

enum StorageKeepalive {
    S3sFs { _fs_root: TempDir },
    RealS3,
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
            _keepalive: StorageKeepalive::S3sFs { _fs_root: fs_root },
        }
    }
}

/// Supertable-shaped backing store (10M warm/cold benches).
pub async fn supertable_storage_fixture() -> StorageFixture {
    backing_store(SUPERTABLE_S3S_BUCKET, "infino-supertable-bench").await
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
        bytes.len() / (1024 * 1024)
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
        env_gib("INFINO_SUPERTABLE_INGEST_CACHE_GIB", 8) * (1u64 << 30),
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
        let b = explicit_gib * (1u64 << 30);
        LOG_ONCE.call_once(|| {
            eprintln!("[tiers] search cache budget = {explicit_gib} GiB (INFINO_SUPERTABLE_SEARCH_CACHE_GIB)");
        });
        b
    } else if let Some(idx) = index_size_bytes.filter(|&s| s > 0) {
        let b = idx + idx / 10;
        LOG_ONCE.call_once(|| {
            eprintln!(
                "[tiers] search cache budget = {:.2} GiB (auto-sized from {:.2} GiB index + 10% headroom)",
                b as f64 / (1u64 << 30) as f64,
                idx as f64 / (1u64 << 30) as f64,
            );
        });
        b
    } else {
        let gib = env_gib("INFINO_SUPERTABLE_INGEST_CACHE_GIB", 8);
        LOG_ONCE.call_once(|| {
            eprintln!("[tiers] search cache budget = {gib} GiB (default)");
        });
        gib * (1u64 << 30)
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
        4 * (1u64 << 30),
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
        cold_fetch_streams: 8,
        cold_fetch_chunk_bytes: 8 * (1u64 << 20),
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
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

#[allow(dead_code)]
pub fn empty_pinned()
-> Arc<dyn Fn() -> HashSet<infino::supertable::manifest::SuperfileUri> + Send + Sync> {
    Arc::new(HashSet::new)
}
