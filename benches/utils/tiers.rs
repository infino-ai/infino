// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared warm / cold storage tier helpers for canonical benches.
//!
//! - **Warm**: `Supertable::open` from object storage + `DiskCacheStore` (local cache hits).
//! - **Cold**: fresh disk cache per iteration → object-store range GETs.
//!
//! Backend is chosen explicitly by `INFINO_BENCH_STORE` (`s3s_fs` default |
//! `s3` | `azure`) — never inferred from which credential is set. `s3` reads
//! `INFINO_REAL_S3_BUCKET`, `azure` reads `INFINO_REAL_AZURE_CONTAINER`.

use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{AzureStorageProvider, S3StorageProvider, StorageProvider};
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
    Warm,
    Cold,
}

impl Tier {
    pub const ALL: [Tier; 2] = [Tier::Warm, Tier::Cold];

    pub fn label(self) -> &'static str {
        match self {
            Tier::Warm => "warm",
            Tier::Cold => "cold",
        }
    }
}

/// Stable report group name for a tiered search bench family (`superfile_vec`, `supertable_fts`, …).
pub fn search_group_name(family: &str, tier: Tier, storage_label: Option<&str>) -> String {
    match tier {
        Tier::Warm => format!("{family}_warm_search"),
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
    /// `true` for a real remote backend (S3 or Azure), `false` for the
    /// in-process s3s-fs emulator.
    pub remote: bool,
    /// Remote prefix to delete when the run finishes (`None` for the
    /// auto-cleaned s3s-fs tempdir, or when `INFINO_BENCH_KEEP_TABLE` is set).
    pub cleanup: Option<PrefixCleanup>,
    _keepalive: StorageKeepalive,
}

enum StorageKeepalive {
    S3sFs { _fs_root: TempDir },
    Remote,
}

/// A real-backend prefix a bench run created and must delete on exit so it
/// accrues no storage cost. The supertable build writes many objects
/// (superfiles, manifests, the pointer) under one unique prefix; cleanup lists
/// every key beneath it and deletes them. `root` is an *un*-prefixed provider
/// (bucket/container root): `list_with_prefix` takes an absolute key prefix
/// and `delete` targets the absolute keys it returns verbatim, so both sides
/// agree on the same keyspace.
#[derive(Clone)]
pub struct PrefixCleanup {
    root: Arc<dyn StorageProvider>,
    prefix: String,
    label: &'static str,
}

/// Delete every object under a bench prefix on its (S3 or Azure) backend.
pub fn cleanup_prefix(cleanup: &PrefixCleanup) {
    let root = Arc::clone(&cleanup.root);
    let prefix = cleanup.prefix.clone();
    let result: Result<usize, String> = block_on(async move {
        let keys = root
            .list_with_prefix(&prefix)
            .await
            .map_err(|e| e.to_string())?;
        let n = keys.len();
        for key in &keys {
            root.delete(key).await.map_err(|e| e.to_string())?;
        }
        Ok(n)
    });
    match result {
        Ok(n) => eprintln!(
            "[tiers] cleanup {} prefix={}: deleted {n} objects",
            cleanup.label, cleanup.prefix
        ),
        Err(e) => eprintln!(
            "[tiers] cleanup {} prefix={}: FAILED ({e}) — objects may remain",
            cleanup.label, cleanup.prefix
        ),
    }
}

/// A single superfile committed to object storage (1M tier benches).
pub struct SuperfileCommitted {
    pub storage: Arc<dyn StorageProvider>,
    pub uri: SuperfileUri,
    /// Object key under the storage provider (same bytes the warm
    /// path built — uploaded verbatim for lazy vector open).
    pub object_path: String,
    pub object_size: u64,
    pub storage_label: &'static str,
    pub cleanup_path: Option<String>,
    _keepalive: StorageKeepalive,
}

impl SuperfileCommitted {
    /// Delete the uploaded object on a real backend. s3s-fs fixtures live
    /// under a tempdir and are cleaned up by dropping `_keepalive`, so they
    /// carry no `cleanup_path` and this is a no-op.
    pub fn cleanup(&self) {
        let Some(path) = self.cleanup_path.as_deref() else {
            return;
        };
        let (storage, label) = (Arc::clone(&self.storage), self.storage_label);
        let result = block_on(async move { storage.delete(path).await });
        match result {
            Ok(()) => eprintln!("[tiers] cleanup {label} superfile path={path}: deleted"),
            Err(e) => eprintln!("[tiers] cleanup {label} superfile path={path}: {e}"),
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

fn azure_container_env() -> Option<String> {
    std::env::var("INFINO_REAL_AZURE_CONTAINER")
        .or_else(|_| std::env::var("INFINO_TEST_REAL_AZURE_CONTAINER"))
        .ok()
}

fn azure_prefix_root(default: &str) -> String {
    std::env::var("INFINO_REAL_AZURE_PREFIX").unwrap_or_else(|_| default.to_string())
}

/// Whether to retain the run's unique prefix instead of deleting it.
fn keep_table() -> bool {
    std::env::var_os("INFINO_BENCH_KEEP_TABLE").is_some()
}

/// The object-store backend a run targets, chosen explicitly by
/// `INFINO_BENCH_STORE` (`s3s_fs` default | `s3` | `azure`) — never inferred
/// from which credential happens to be set.
#[derive(Debug, PartialEq, Eq)]
enum Backend {
    /// In-process s3s-fs emulator (no creds, no network).
    S3sFs,
    /// Real AWS S3.
    S3 { bucket: String },
    /// Real Azure Blob.
    Azure { container: String },
}

impl Backend {
    fn from_env() -> Result<Self, String> {
        let store = std::env::var("INFINO_BENCH_STORE").unwrap_or_else(|_| "s3s_fs".into());
        Self::parse(&store, real_s3_bucket_env(), azure_container_env())
    }

    /// Pure resolution: real backends require their location env. Split out
    /// so selection is unit-testable without mutating process env.
    fn parse(
        store: &str,
        s3_bucket: Option<String>,
        azure_container: Option<String>,
    ) -> Result<Self, String> {
        match store {
            "s3s_fs" => Ok(Self::S3sFs),
            "s3" => s3_bucket
                .map(|bucket| Self::S3 { bucket })
                .ok_or_else(|| "INFINO_BENCH_STORE=s3 requires INFINO_REAL_S3_BUCKET".to_string()),
            "azure" => azure_container
                .map(|container| Self::Azure { container })
                .ok_or_else(|| {
                    "INFINO_BENCH_STORE=azure requires INFINO_REAL_AZURE_CONTAINER".to_string()
                }),
            other => Err(format!(
                "unknown INFINO_BENCH_STORE={other} (want s3s_fs|s3|azure)"
            )),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::S3sFs => "s3s_fs",
            Self::S3 { .. } => "s3",
            Self::Azure { .. } => "azure",
        }
    }

    /// Namespace root under the bucket/container for this run's objects.
    fn prefix_root(&self, default: &str) -> String {
        match self {
            Self::S3sFs => default.to_string(),
            Self::S3 { .. } => real_s3_prefix_root(default),
            Self::Azure { .. } => azure_prefix_root(default),
        }
    }

    /// Provider for a real backend, scoped to `prefix`. `None` for s3s-fs
    /// (spawned separately). `prefix = ""` gives the bucket/container root.
    fn provider(&self, prefix: &str) -> Option<Arc<dyn StorageProvider>> {
        match self {
            Self::S3sFs => None,
            Self::S3 { bucket } => Some(Arc::new(
                S3StorageProvider::new_with_prefix(bucket, prefix).expect("real S3 provider"),
            )),
            Self::Azure { container } => Some(Arc::new(
                AzureStorageProvider::new_with_prefix(container, prefix)
                    .expect("real Azure provider"),
            )),
        }
    }
}

/// Pre-flight for the supertable bench: it needs a real object store (`s3`
/// or `azure`) for the multi-commit OCC. `Ok` when one is selected + ready,
/// `Err(reason)` otherwise (default s3s-fs, missing creds, unknown store).
pub fn supertable_backend_check() -> Result<(), String> {
    match Backend::from_env()? {
        Backend::S3sFs => Err(SUPERTABLE_REQUIRES_REAL_OBJECT_STORE.to_string()),
        _ => Ok(()),
    }
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

/// Build the fixture for a real backend (S3 or Azure): a unique prefix, a
/// prefix-scoped provider, and — unless `INFINO_BENCH_KEEP_TABLE` — a cleanup
/// over that prefix on the bucket/container root.
fn remote_fixture(backend: Backend, prefix_default: &str) -> StorageFixture {
    let label = backend.label();
    let prefix = unique_bench_prefix(&backend.prefix_root(prefix_default));
    let storage = backend.provider(&prefix).expect("remote provider");
    eprintln!("[tiers] {label} prefix={prefix}");
    let cleanup = if keep_table() {
        eprintln!(
            "[tiers] keeping {label} prefix={prefix} (INFINO_BENCH_KEEP_TABLE; cleanup skipped)"
        );
        None
    } else {
        Some(PrefixCleanup {
            root: backend.provider("").expect("remote root provider"),
            prefix,
            label,
        })
    };
    StorageFixture {
        storage,
        storage_label: label,
        remote: true,
        cleanup,
        _keepalive: StorageKeepalive::Remote,
    }
}

async fn backing_store(s3s_bucket: &str, prefix_default: &str) -> StorageFixture {
    let backend = Backend::from_env().unwrap_or_else(|e| panic!("{e}"));
    let Backend::S3sFs = backend else {
        return remote_fixture(backend, prefix_default);
    };
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
         ##  WARNING: INFINO_BENCH_STORE=s3s_fs — in-process emulator, NOT a real store.##\n\
         ##  It reproduces request count and byte volume, not network latency, so       ##\n\
         ##  warm/cold timings here are not representative. Set INFINO_BENCH_STORE=s3    ##\n\
         ##  (+ INFINO_REAL_S3_BUCKET) or =azure (+ INFINO_REAL_AZURE_CONTAINER) for real.##\n\
         ################################################################################\n\
         [tiers] s3s-fs endpoint={endpoint}  storage_label=s3s_fs  (NOT a real store)\n"
    );
    StorageFixture {
        storage,
        storage_label: "s3s_fs",
        remote: false,
        cleanup: None,
        _keepalive: StorageKeepalive::S3sFs { _fs_root: fs_root },
    }
}

/// Error string for the supertable backend guard. A constant so the `run()`
/// pre-flight ([`supertable_backend_check`]) and this fixture agree.
pub const SUPERTABLE_REQUIRES_REAL_OBJECT_STORE: &str = "\
the supertable object-store bench requires a real object store. Set \
INFINO_BENCH_STORE=s3 (+ INFINO_REAL_S3_BUCKET + AWS creds) or =azure (+ \
INFINO_REAL_AZURE_CONTAINER + AZURE_STORAGE_ACCOUNT_NAME/_KEY). The s3s-fs \
emulator is not usable here: it does not implement conditional If-Match PUTs, \
which the supertable's multi-commit OCC requires, so every commit after the \
first would lose the CAS.";

/// Supertable-shaped backing store (multi-superfile, multi-commit benches).
///
/// **Real object store only** (S3 or Azure). The supertable build commits
/// many times, so its OCC pointer update rides on the conditional `If-Match`
/// PUT. The in-process `s3s-fs` emulator does not implement those (every
/// commit after the first loses the CAS), so this fixture rejects it and
/// panics with [`SUPERTABLE_REQUIRES_REAL_OBJECT_STORE`]. The returned fixture
/// carries a [`PrefixCleanup`] so the caller can delete the unique prefix when
/// the run finishes.
pub async fn supertable_storage_fixture() -> StorageFixture {
    match Backend::from_env().unwrap_or_else(|e| panic!("{e}")) {
        Backend::S3sFs => panic!("{SUPERTABLE_REQUIRES_REAL_OBJECT_STORE}"),
        backend => remote_fixture(backend, "infino-supertable-bench"),
    }
}

/// Storage for a prepared dataset at a fixed prefix — no unique suffix, no
/// cleanup, so the data persists across runs. `subdir` namespaces the modality
/// under the base prefix from `INFINO_BENCH_DATASET_PREFIX`. Real backend only,
/// same as [`supertable_storage_fixture`].
pub async fn dataset_storage_fixture(subdir: &str) -> StorageFixture {
    let backend = match Backend::from_env().unwrap_or_else(|e| panic!("{e}")) {
        Backend::S3sFs => panic!("{SUPERTABLE_REQUIRES_REAL_OBJECT_STORE}"),
        backend => backend,
    };
    let base = crate::dataset::dataset_prefix()
        .expect("dataset_storage_fixture requires INFINO_BENCH_DATASET_PREFIX");
    let prefix = format!("{}/{subdir}", base.trim_matches('/'));
    let label = backend.label();
    let storage = backend.provider(&prefix).expect("dataset provider");
    eprintln!("[tiers] dataset {label} prefix={prefix}");
    StorageFixture {
        storage,
        storage_label: label,
        remote: true,
        cleanup: None,
        _keepalive: StorageKeepalive::Remote,
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
        // Delete the uploaded object on a real backend, unless asked to keep it.
        cleanup_path: (fixture.remote && !keep_table()).then_some(path),
        _keepalive: fixture._keepalive,
    }
}

/// Backing object store for superfile-shaped warm/cold benches (1M).
///
/// Unlike [`supertable_storage_fixture`], this allows the default `s3s_fs`
/// emulator because superfile-shaped benches do not rely on multi-commit OCC.
pub async fn superfile_storage_fixture() -> StorageFixture {
    backing_store(SUPERFILE_S3S_BUCKET, "infino-superfile-bench").await
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
/// Ingest attaches this cache only to keep superfile bytes out of the
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
///    size from the manifest — ensures the warm bench is truly warm.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_s3s_fs() {
        assert_eq!(Backend::parse("s3s_fs", None, None), Ok(Backend::S3sFs));
    }

    #[test]
    fn parse_s3_needs_bucket() {
        assert_eq!(
            Backend::parse("s3", Some("bkt".into()), None),
            Ok(Backend::S3 {
                bucket: "bkt".into()
            })
        );
        assert!(Backend::parse("s3", None, None).is_err());
    }

    #[test]
    fn parse_azure_needs_container() {
        assert_eq!(
            Backend::parse("azure", None, Some("c".into())),
            Ok(Backend::Azure {
                container: "c".into()
            })
        );
        assert!(Backend::parse("azure", None, None).is_err());
    }

    #[test]
    fn parse_does_not_infer_from_creds() {
        // Creds present but store=s3s_fs → still the emulator.
        assert_eq!(
            Backend::parse("s3s_fs", Some("bkt".into()), Some("c".into())),
            Ok(Backend::S3sFs)
        );
    }

    #[test]
    fn parse_rejects_unknown_store() {
        assert!(Backend::parse("gcs", None, None).is_err());
    }

    #[test]
    fn labels_match_backend() {
        assert_eq!(Backend::S3sFs.label(), "s3s_fs");
        assert_eq!(Backend::S3 { bucket: "b".into() }.label(), "s3");
        assert_eq!(
            Backend::Azure {
                container: "c".into()
            }
            .label(),
            "azure"
        );
    }
}
