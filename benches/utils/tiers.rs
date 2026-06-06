//! Shared hot / cold storage tier helpers for canonical benches.
//!
//! - **Hot**: `Supertable::open` from object storage + `DiskCacheStore` (local cache hits).
//! - **Cold**: fresh disk cache per iteration → object-store range GETs.
//!
//! Backend is chosen explicitly via `INFINO_BENCH_STORE` (`s3s_fs` default
//! | `s3` | `azure`); `s3` reads `INFINO_REAL_S3_BUCKET`, `azure` reads
//! `INFINO_REAL_AZURE_CONTAINER`. Never inferred from which credential is
//! set.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};

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

/// Storage labels that can appear in a warm/cold search group name
/// (`{family}_{tier}_search_{label}`). The emulator is `s3s_fs`; real
/// backends use [`RemoteBackend::label`]. Markdown report generation
/// iterates this set; a unit test guards it against `RemoteBackend::label`.
pub const STORAGE_LABELS: &[&str] = &["s3s_fs", "s3", "azure"];

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
    _keepalive: StorageKeepalive,
}

enum StorageKeepalive {
    S3sFs { _fs_root: TempDir },
    Remote,
}

/// A single superfile committed to object storage (1M tier benches).
pub struct SuperfileCommitted {
    pub storage: Arc<dyn StorageProvider>,
    pub uri: SuperfileUri,
    pub storage_label: &'static str,
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

/// A remote prefix to delete when the process ends. Registered only for
/// ephemeral runs (no `INFINO_BENCH_DATASET`): each such run writes a
/// throwaway table under a unique prefix and must not leave it behind.
struct CleanupTarget {
    /// Provider scoped to the bucket root, so `prefix` is listed/deleted
    /// as an absolute path.
    root: Arc<dyn StorageProvider>,
    prefix: String,
    label: &'static str,
}

static CLEANUP: Mutex<Vec<CleanupTarget>> = Mutex::new(Vec::new());

/// Delete every remote prefix registered this run. Call once from the
/// `main` of any bench that goes through `backing_store` against a real
/// store, after the criterion groups finish — statics don't run `Drop`
/// at process exit, so teardown is explicit. No-op for s3s-fs (its
/// tempdir self-cleans) and for persisted runs (nothing is registered).
pub fn cleanup_ephemeral() {
    let targets = std::mem::take(&mut *CLEANUP.lock().expect("cleanup registry"));
    if targets.is_empty() {
        return;
    }
    block_on(async {
        for t in targets {
            let keys = match t.root.list_with_prefix(&t.prefix).await {
                Ok(keys) => keys,
                // Surface rather than swallow: a failed list means the
                // prefix may be leaking, which is worth seeing.
                Err(e) => {
                    eprintln!(
                        "[tiers] cleanup list failed for {} ({}): {e}",
                        t.prefix, t.label
                    );
                    continue;
                }
            };
            for key in &keys {
                let _ = t.root.delete(key).await;
            }
            eprintln!(
                "[tiers] cleaned {} objects under {} ({})",
                keys.len(),
                t.prefix,
                t.label
            );
        }
    });
}

pub fn real_s3_bucket_env() -> Option<String> {
    std::env::var("INFINO_REAL_S3_BUCKET")
        .or_else(|_| std::env::var("INFINO_TEST_REAL_S3_BUCKET"))
        .ok()
}

pub fn real_s3_prefix_root(default: &str) -> String {
    std::env::var("INFINO_REAL_S3_PREFIX").unwrap_or_else(|_| default.to_string())
}

fn real_azure_container_env() -> Option<String> {
    std::env::var("INFINO_REAL_AZURE_CONTAINER")
        .or_else(|_| std::env::var("INFINO_TEST_REAL_AZURE_CONTAINER"))
        .ok()
}

/// Object store the warm/cold benches run against. Chosen **explicitly**
/// via `INFINO_BENCH_STORE` — never inferred from which credential happens
/// to be exported.
#[derive(Debug, PartialEq, Eq)]
enum BenchStore {
    /// In-process s3s-fs emulator (default; no credentials, no network).
    S3sFs,
    /// A real remote object store.
    Remote(RemoteBackend),
}

impl BenchStore {
    /// Resolve from `INFINO_BENCH_STORE` (`s3s_fs` default | `s3` | `azure`).
    fn from_env() -> Self {
        let store = std::env::var("INFINO_BENCH_STORE").unwrap_or_else(|_| "s3s_fs".into());
        Self::parse(&store, real_s3_bucket_env(), real_azure_container_env())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Pure resolution — the chosen backend must have its location env set.
    /// Split out so selection is unit-testable without mutating process env.
    fn parse(
        store: &str,
        s3_bucket: Option<String>,
        azure_container: Option<String>,
    ) -> Result<Self, String> {
        match store {
            "s3s_fs" => Ok(Self::S3sFs),
            "s3" => s3_bucket
                .map(|bucket| Self::Remote(RemoteBackend::S3 { bucket }))
                .ok_or_else(|| "INFINO_BENCH_STORE=s3 requires INFINO_REAL_S3_BUCKET".to_string()),
            "azure" => azure_container
                .map(|container| Self::Remote(RemoteBackend::Azure { container }))
                .ok_or_else(|| {
                    "INFINO_BENCH_STORE=azure requires INFINO_REAL_AZURE_CONTAINER".to_string()
                }),
            other => Err(format!(
                "unknown INFINO_BENCH_STORE={other} (want s3s_fs|s3|azure)"
            )),
        }
    }
}

/// A real (non-emulator) object store for warm/cold tiers.
#[derive(Debug, PartialEq, Eq)]
enum RemoteBackend {
    S3 { bucket: String },
    Azure { container: String },
}

impl RemoteBackend {
    fn label(&self) -> &'static str {
        match self {
            Self::S3 { .. } => "s3",
            Self::Azure { .. } => "azure",
        }
    }

    /// Namespace root for this run's objects: a per-backend env override,
    /// else `default`.
    fn prefix_root(&self, default: &str) -> String {
        match self {
            Self::S3 { .. } => real_s3_prefix_root(default),
            Self::Azure { .. } => {
                std::env::var("INFINO_REAL_AZURE_PREFIX").unwrap_or_else(|_| default.to_string())
            }
        }
    }

    /// Provider scoped to `prefix`. The single per-backend construction
    /// site; adding a backend is one arm here.
    fn provider(&self, prefix: &str) -> Arc<dyn StorageProvider> {
        match self {
            Self::S3 { bucket } => Arc::new(
                S3StorageProvider::new_with_prefix(bucket, prefix).expect("real S3 provider"),
            ),
            Self::Azure { container } => Arc::new(
                AzureStorageProvider::new_with_prefix(container, prefix)
                    .expect("real Azure provider"),
            ),
        }
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

async fn backing_store(s3s_bucket: &str, prefix_default: &str) -> StorageFixture {
    // s3s-fs returns directly; the remote backends share the construction below.
    let backend = match BenchStore::from_env() {
        BenchStore::Remote(backend) => backend,
        BenchStore::S3sFs => {
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
                 ##  WARNING: benchmarking against the s3s-fs emulator, NOT a real object store.##\n\
                 ##  The emulator reproduces request count and byte volume, not network         ##\n\
                 ##  latency, so warm/cold timings here are not representative.                  ##\n\
                 ##  Set INFINO_BENCH_STORE=s3|azure (+ the backend's container env) for real.   ##\n\
                 ################################################################################\n\
                 [tiers] s3s-fs endpoint={endpoint}  storage_label=s3s_fs  (NOT a real store)\n"
            );
            return StorageFixture {
                storage,
                storage_label: "s3s_fs",
                _keepalive: StorageKeepalive::S3sFs { _fs_root: fs_root },
            };
        }
    };

    let prefix = unique_bench_prefix(&backend.prefix_root(prefix_default));
    let storage = backend.provider(&prefix);
    eprintln!("[tiers] {} prefix={prefix}", backend.label());

    // Ephemeral runs (no persisted dataset) own this unique prefix and must
    // delete it at the end; persisted runs (`INFINO_BENCH_DATASET`) keep the
    // table for reuse, so they register nothing.
    if std::env::var("INFINO_BENCH_DATASET").is_err() {
        CLEANUP
            .lock()
            .expect("cleanup registry")
            .push(CleanupTarget {
                root: backend.provider(""),
                prefix: prefix.clone(),
                label: backend.label(),
            });
    }

    StorageFixture {
        storage,
        storage_label: backend.label(),
        _keepalive: StorageKeepalive::Remote,
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
        storage_label: fixture.storage_label,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_to_emulator() {
        assert_eq!(
            BenchStore::parse("s3s_fs", None, None),
            Ok(BenchStore::S3sFs)
        );
    }

    #[test]
    fn parse_s3_needs_bucket() {
        assert_eq!(
            BenchStore::parse("s3", Some("bkt".into()), None),
            Ok(BenchStore::Remote(RemoteBackend::S3 {
                bucket: "bkt".into()
            }))
        );
        assert!(BenchStore::parse("s3", None, None).is_err());
    }

    #[test]
    fn parse_azure_needs_container() {
        assert_eq!(
            BenchStore::parse("azure", None, Some("c".into())),
            Ok(BenchStore::Remote(RemoteBackend::Azure {
                container: "c".into()
            }))
        );
        assert!(BenchStore::parse("azure", None, None).is_err());
    }

    #[test]
    fn parse_rejects_unknown_store() {
        assert!(BenchStore::parse("gcs", None, None).is_err());
    }

    #[test]
    fn parse_does_not_infer_from_creds() {
        // Both creds present but store=s3s_fs → still the emulator. No
        // backend is ever picked from which credential happens to be set.
        assert_eq!(
            BenchStore::parse("s3s_fs", Some("bkt".into()), Some("c".into())),
            Ok(BenchStore::S3sFs)
        );
    }

    #[test]
    fn label_matches_backend() {
        assert_eq!(
            RemoteBackend::Azure {
                container: "c".into()
            }
            .label(),
            "azure"
        );
        assert_eq!(RemoteBackend::S3 { bucket: "b".into() }.label(), "s3");
    }

    #[test]
    fn storage_labels_cover_every_remote_backend() {
        // STORAGE_LABELS is the single source markdown/report code iterates;
        // every real backend's label must be in it (plus the emulator).
        for label in [
            RemoteBackend::S3 { bucket: "b".into() }.label(),
            RemoteBackend::Azure {
                container: "c".into(),
            }
            .label(),
        ] {
            assert!(
                STORAGE_LABELS.contains(&label),
                "{label} missing from STORAGE_LABELS"
            );
        }
        assert!(STORAGE_LABELS.contains(&"s3s_fs"));
    }
}
