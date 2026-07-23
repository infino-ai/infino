// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Local RustFS daemon lifecycle for benchmark object-store fixtures.
//!
//! Spawns a child `rustfs` process with bench-generated TLS credentials,
//! waits for `/health`, creates the target bucket, and builds an HTTPS
//! `S3StorageProvider` that trusts the bench-local CA.
//!
//! ## Session model (tests and benches)
//!
//! [`session`] starts one RustFS daemon per process on **first use** (`OnceLock`);
//! the child handle is owned by the session so it outlives individual fixtures.
//! Tests
//! should open storage via [`open_test_fixture`] — no explicit session create or
//! teardown. Buckets are isolated per fixture; they are not deleted on drop (the
//! process exit tears down the daemon). For explicit bucket cleanup tests, use
//! [`RustFsSession::open_unique_bucket`] and [`release_lease`].
//!
//! Auto-downloaded binaries are cached under `target/infino-bench/rustfs/` with a
//! `rustfs.version` stamp. When the stamp is older than 24 hours, it is compared
//! to [`DEFAULT_RUSTFS_VERSION`] (or `INFINO_RUSTFS_VERSION`); a mismatch removes
//! the stale binary and fetches the pinned release.

use std::{
    io::{Cursor, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime},
};

use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair, SerialNumber};
use reqwest::header::CONTENT_LENGTH;
use tempfile::TempDir;

use crate::rss::fmt_bytes;

/// Default pinned RustFS release for auto-download.
const DEFAULT_RUSTFS_VERSION: &str = "1.0.0-alpha.90";
/// S3 region passed to the object_store client for local RustFS.
const RUSTFS_REGION: &str = "us-east-1";
/// Default bucket for RustFS bench fixtures.
pub const RUSTFS_BENCH_BUCKET: &str = "infino-bench";
/// Milliseconds between health polls while RustFS starts.
const HEALTH_POLL_INTERVAL_MS: u64 = 200;
/// Maximum time to wait for RustFS `/health` after spawn.
const HEALTH_TIMEOUT_SECS: u64 = 60;
/// Grace period after SIGKILL before a second kill attempt during teardown.
const TEARDOWN_GRACE_MS: u64 = 2_000;
/// Poll interval while waiting for a killed RustFS child to exit.
const TEARDOWN_POLL_MS: u64 = 50;
/// Spawn attempts when the chosen loopback port is already in use or health check fails.
const RUSTFS_SPAWN_MAX_ATTEMPTS: u32 = 5;
/// Inclusive lower bound of the IANA ephemeral port range for random loopback picks.
const EPHEMERAL_PORT_MIN: u16 = 49_152;
/// Inclusive upper bound of the IANA ephemeral port range for random loopback picks.
const EPHEMERAL_PORT_MAX: u16 = 65_535;
/// Filename of the upstream checksum manifest on RustFS GitHub releases.
const RUSTFS_SHA256SUMS_ASSET: &str = "SHA256SUMS";
/// Maximum list-and-delete passes while emptying a leased bucket before DeleteBucket.
const BUCKET_EMPTY_MAX_PASSES: u32 = 10;
/// Poll interval while waiting for a background RustFS binary download.
const BINARY_PREFETCH_POLL_MS: u64 = 50;
/// Re-check the pinned RustFS version against `rustfs.version` only after this
/// interval. Fresh stamps skip the string compare (and any re-download) to avoid
/// network I/O on every test run.
const RUSTFS_CACHE_VERSION_RECHECK_SECS: u64 = 24 * 60 * 60;

enum BinaryPrefetchState {
    Idle,
    Running,
    Ready(Result<PathBuf, String>),
}

static BINARY_PREFETCH: Mutex<BinaryPrefetchState> = Mutex::new(BinaryPrefetchState::Idle);

struct S3SignParams<'a> {
    method: &'a str,
    canonical_uri: &'a str,
    /// Sorted, URI-encoded query string without leading `?` (empty when none).
    canonical_query: &'a str,
    host: &'a str,
    amz_date: &'a str,
    payload_hash: &'a str,
    access_key: &'a str,
    secret_key: &'a str,
    region: &'a str,
}
/// Length of generated access/secret keys when env overrides are absent.
const GENERATED_KEY_LEN: usize = 20;
const GENERATED_SECRET_LEN: usize = 40;
/// Allowed characters for generated RustFS credentials.
const GENERATED_KEY_CHARSET: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/// Monotonic suffix for unique RustFS bucket names within one process.
const RUSTFS_BUCKET_COUNTER_START: u64 = 0;

static RUSTFS_BUCKET_COUNTER: AtomicU64 = AtomicU64::new(RUSTFS_BUCKET_COUNTER_START);

/// Running RustFS child plus tempdirs that must outlive the process.
pub struct RustFsHandle {
    pub endpoint: String,
    pub bucket: String,
    pub access_key: String,
    pub secret_key: String,
    pub ca_pem: Vec<u8>,
    pub addr: SocketAddr,
    child: Child,
    _data_dir: TempDir,
    _tls_dir: TempDir,
}

impl Drop for RustFsHandle {
    fn drop(&mut self) {
        terminate_child(&mut self.child);
    }
}

/// Send `kill` to a spawned RustFS child, poll until exit or grace elapses, then
/// `kill` again and `wait` (no `SIGTERM` — RustFS teardown is best-effort).
pub fn terminate_child(child: &mut Child) {
    terminate_child_impl(child);
}

fn terminate_child_impl(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    let _ = child.kill();
    let deadline = Instant::now() + Duration::from_millis(TEARDOWN_GRACE_MS);
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(TEARDOWN_POLL_MS));
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// When set, skip RustFS-backed wire/integration tests (offline hosts, unsupported
/// platforms without `INFINO_RUSTFS_BIN`, etc.).
pub fn rustfs_tests_disabled() -> bool {
    std::env::var_os("INFINO_TEST_DISABLE_RUSTFS").is_some()
}

/// Whether this host can auto-download a RustFS release binary.
pub fn rustfs_auto_download_supported() -> bool {
    release_asset_name(&rustfs_version()).is_ok()
}

/// Whether RustFS wire tests should run on this host (not disabled and binary is
/// available or can be fetched).
pub fn rustfs_tests_runnable() -> bool {
    if rustfs_tests_disabled() {
        return false;
    }
    if std::env::var("INFINO_RUSTFS_BIN").is_ok() {
        return true;
    }
    if rustfs_cache_valid() {
        return true;
    }
    if which_rustfs_on_path().is_some() {
        return true;
    }
    rustfs_auto_download_supported()
}

fn rustfs_cache_valid() -> bool {
    rustfs_cache_usable().is_some()
}

fn rustfs_binary_located_sync() -> Option<PathBuf> {
    rustfs_cache_usable().or_else(which_rustfs_on_path)
}

/// Use the auto-downloaded cache when the binary exists and the version stamp is
/// fresh, or when the stamp is older but still matches the pinned release.
fn rustfs_cache_usable() -> Option<PathBuf> {
    let cached = rustfs_cache_binary_path();
    if !cached.is_file() {
        return None;
    }
    if read_rustfs_cache_version().is_none() {
        log_rustfs_progress(
            "[rustfs] cached binary has no version stamp — removing and re-downloading",
        );
        clear_rustfs_cache();
        return None;
    }
    if rustfs_version_stamp_within_recheck_window() {
        return Some(cached);
    }
    let have = read_rustfs_cache_version().expect("version stamp checked above");
    let want = rustfs_version();
    if have == want {
        return Some(cached);
    }
    log_rustfs_progress(&format!(
        "[rustfs] cached binary is {have}, want {want} — removing stale cache"
    ));
    clear_rustfs_cache();
    None
}

fn download_rustfs_binary_sync() -> Result<PathBuf, String> {
    let cached = rustfs_cache_binary_path();
    download_rustfs_binary(&cached)?;
    Ok(cached)
}

/// Start locating or downloading the RustFS binary on a background thread.
///
/// No-op when tests are disabled or the binary is already on disk / `PATH`.
/// [`ensure_rustfs_binary`] waits for an in-flight prefetch before blocking itself.
pub fn prefetch_rustfs_binary() {
    if rustfs_tests_disabled() || rustfs_binary_located_sync().is_some() {
        return;
    }
    let mut state = BINARY_PREFETCH.lock().expect("rustfs binary prefetch lock");
    match *state {
        BinaryPrefetchState::Idle => {
            *state = BinaryPrefetchState::Running;
            std::thread::spawn(|| {
                let result = download_rustfs_binary_sync();
                *BINARY_PREFETCH.lock().expect("rustfs binary prefetch lock") =
                    BinaryPrefetchState::Ready(result);
            });
        }
        BinaryPrefetchState::Running | BinaryPrefetchState::Ready(_) => {}
    }
}

fn wait_for_binary_prefetch() -> Result<PathBuf, String> {
    loop {
        let mut state = BINARY_PREFETCH.lock().expect("rustfs binary prefetch lock");
        match std::mem::replace(&mut *state, BinaryPrefetchState::Idle) {
            BinaryPrefetchState::Idle => return download_rustfs_binary_sync(),
            BinaryPrefetchState::Running => {
                *state = BinaryPrefetchState::Running;
                drop(state);
                std::thread::sleep(Duration::from_millis(BINARY_PREFETCH_POLL_MS));
            }
            BinaryPrefetchState::Ready(Ok(path)) => {
                *state = BinaryPrefetchState::Ready(Ok(path.clone()));
                return Ok(path);
            }
            BinaryPrefetchState::Ready(Err(err)) => {
                // Leave Idle so a later call can retry a transient download failure.
                return Err(err);
            }
        }
    }
}

/// Locate or download the `rustfs` binary.
pub fn ensure_rustfs_binary() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("INFINO_RUSTFS_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "INFINO_RUSTFS_BIN={} is not a file",
            path.display()
        ));
    }
    if let Some(path) = rustfs_binary_located_sync() {
        return Ok(path);
    }
    prefetch_rustfs_binary();
    wait_for_binary_prefetch()
}

/// Spawn RustFS on a random loopback port with HTTPS enabled (no bucket yet).
fn spawn_rustfs_daemon() -> Result<RustFsHandle, String> {
    let binary = ensure_rustfs_binary()?;
    let (access_key, secret_key) = rustfs_credentials();
    let data_dir = TempDir::new().map_err(|e| e.to_string())?;
    let (tls_dir, ca_pem) = generate_tls_material()?;

    let mut last_err = String::new();
    for attempt in 1..=RUSTFS_SPAWN_MAX_ATTEMPTS {
        let addr = random_loopback_addr();
        let port = addr.port();
        let address = format!("127.0.0.1:{port}");
        let endpoint = format!("https://{address}");

        let mut child = Command::new(&binary)
            .arg(data_dir.path())
            .env("RUSTFS_ACCESS_KEY", &access_key)
            .env("RUSTFS_SECRET_KEY", &secret_key)
            .env("RUSTFS_VOLUMES", data_dir.path())
            .env("RUSTFS_ADDRESS", &address)
            .env("RUSTFS_TLS_PATH", tls_dir.path())
            .env("RUSTFS_CONSOLE_ENABLE", "false")
            .env("RUST_LOG", "warn")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to spawn rustfs at {}: {e}", binary.display()))?;

        if child_exited(&mut child) {
            last_err = format!("rustfs exited immediately on port {port}");
            eprintln!("[rustfs] spawn attempt {attempt}/{RUSTFS_SPAWN_MAX_ATTEMPTS}: {last_err}");
            continue;
        }

        match wait_for_health(&mut child, &endpoint, &ca_pem) {
            Ok(()) => {
                eprintln!("[rustfs] endpoint={endpoint} storage_label=rustfs");
                return Ok(RustFsHandle {
                    endpoint,
                    bucket: String::new(),
                    access_key,
                    secret_key,
                    ca_pem,
                    addr,
                    child,
                    _data_dir: data_dir,
                    _tls_dir: tls_dir,
                });
            }
            Err(e) => {
                terminate_child(&mut child);
                last_err = e;
                eprintln!(
                    "[rustfs] spawn attempt {attempt}/{RUSTFS_SPAWN_MAX_ATTEMPTS} on port {port}: {last_err}"
                );
            }
        }
    }

    Err(format!(
        "rustfs failed to start after {RUSTFS_SPAWN_MAX_ATTEMPTS} attempts: {last_err}"
    ))
}

/// Spawn a dedicated RustFS child and create `bucket` (legacy standalone fixture).
///
/// Prefer [`open_test_fixture`] for tests — it uses the lazy shared [`session`].
#[deprecated(note = "use open_test_fixture(); session() is lazy-loaded on first use")]
pub fn spawn_rustfs(bucket: &str) -> Result<RustFsHandle, String> {
    let mut handle = spawn_rustfs_daemon()?;
    provision_bucket(
        &handle.endpoint,
        &handle.access_key,
        &handle.secret_key,
        &handle.ca_pem,
        bucket,
    )?;
    handle.bucket = bucket.to_string();
    eprintln!("[rustfs] bucket={bucket}");
    Ok(handle)
}

/// Connection metadata for a long-lived RustFS daemon shared across a process.
pub struct RustFsSession {
    endpoint: String,
    access_key: String,
    secret_key: String,
    ca_pem: Vec<u8>,
    _handle: RustFsHandle,
}

impl RustFsSession {
    /// HTTPS endpoint (`https://127.0.0.1:<port>`) for the shared session daemon.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Access key the daemon accepts for SigV4.
    pub fn access_key(&self) -> &str {
        &self.access_key
    }

    /// Secret key paired with [`Self::access_key`].
    pub fn secret_key(&self) -> &str {
        &self.secret_key
    }

    /// Bench-local CA PEM used to trust the daemon's TLS certificate.
    pub fn ca_pem(&self) -> &[u8] {
        &self.ca_pem
    }
}

static RUSTFS_SESSION: OnceLock<Arc<RustFsSession>> = OnceLock::new();
static RUSTFS_SESSION_INIT: Mutex<()> = Mutex::new(());

/// One RustFS daemon per process; lazy-loaded on first call. The child outlives
/// all fixtures (see [`open_test_fixture`]).
///
/// Transient spawn/download failures are not cached: only a successful session
/// is stored, so a later call can retry.
pub fn session() -> Result<Arc<RustFsSession>, String> {
    if let Some(session) = RUSTFS_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let _guard = RUSTFS_SESSION_INIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(session) = RUSTFS_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let handle = spawn_rustfs_daemon()?;
    let session = Arc::new(RustFsSession {
        endpoint: handle.endpoint.clone(),
        access_key: handle.access_key.clone(),
        secret_key: handle.secret_key.clone(),
        ca_pem: handle.ca_pem.clone(),
        _handle: handle,
    });
    let _ = RUSTFS_SESSION.set(Arc::clone(&session));
    Ok(session)
}

/// Scoped bucket on the shared session. Empties and deletes the bucket on drop
/// unless `INFINO_BENCH_KEEP_TABLE` is set or the lease was opened persistent.
pub struct RustFsBucketLease {
    pub bucket: String,
    pub storage: Arc<dyn StorageProvider>,
    session: Arc<RustFsSession>,
    cleanup_on_drop: bool,
}

impl RustFsBucketLease {
    /// Lease a bucket that survives drop (dataset / retained-prefix workflows).
    pub fn persistent(
        session: Arc<RustFsSession>,
        bucket: &str,
        prefix: &str,
    ) -> Result<Self, String> {
        session.open_bucket(bucket, prefix, false)
    }
}

impl Drop for RustFsBucketLease {
    fn drop(&mut self) {
        if !self.cleanup_on_drop {
            return;
        }
        if keep_rustfs_bucket() {
            eprintln!(
                "[rustfs] keeping bucket={} (INFINO_BENCH_KEEP_TABLE)",
                self.bucket
            );
            return;
        }
        match empty_and_delete_bucket(&self.session, &self.bucket) {
            Ok(()) => eprintln!("[rustfs] cleanup bucket={}: deleted", self.bucket),
            Err(e) => eprintln!(
                "[rustfs] cleanup bucket={}: FAILED ({e}) — objects may remain",
                self.bucket
            ),
        }
    }
}

impl RustFsSession {
    /// Create `bucket` and return a provider scoped to `prefix`.
    pub fn open_bucket(
        self: &Arc<Self>,
        bucket: &str,
        prefix: &str,
        cleanup_on_drop: bool,
    ) -> Result<RustFsBucketLease, String> {
        provision_bucket(
            &self.endpoint,
            &self.access_key,
            &self.secret_key,
            &self.ca_pem,
            bucket,
        )?;
        let storage = build_rustfs_provider(
            &self.endpoint,
            bucket,
            prefix,
            &self.access_key,
            &self.secret_key,
            &self.ca_pem,
        )?;
        eprintln!("[rustfs] bucket={bucket} prefix={prefix}");
        Ok(RustFsBucketLease {
            bucket: bucket.to_string(),
            storage,
            session: Arc::clone(self),
            cleanup_on_drop,
        })
    }

    /// Fresh bucket name (`infino-{pid}-{nanos}`) for an isolated bench run.
    pub fn open_unique_bucket(self: &Arc<Self>, prefix: &str) -> Result<RustFsBucketLease, String> {
        self.open_bucket(&unique_rustfs_bucket_name(), prefix, true)
    }

    /// Unique bucket for tests: same as [`open_unique_bucket`] but leaves the
    /// bucket on the daemon when the lease drops (no blocking cleanup).
    pub fn open_test_bucket(self: &Arc<Self>, prefix: &str) -> Result<RustFsBucketLease, String> {
        self.open_bucket(&unique_rustfs_bucket_name(), prefix, false)
    }
}

/// Storage + bucket lease for a RustFS-backed test.
///
/// Open with [`open_test_fixture`]. The shared session daemon is started lazily on
/// first use. Dropping this fixture does not tear down the daemon or delete the
/// bucket (safe from async tests).
pub struct RustFsTestFixture {
    pub storage: Arc<dyn StorageProvider>,
    pub bucket: String,
}

/// Returns `true` when RustFS tests should run on this host.
pub fn begin_rustfs_test(test_name: &str) -> bool {
    prefetch_rustfs_binary();
    if rustfs_tests_disabled() {
        eprintln!("{test_name}: skipped (INFINO_TEST_DISABLE_RUSTFS is set)");
        return false;
    }
    if !rustfs_tests_runnable() {
        eprintln!("{test_name}: skipped (RustFS binary unavailable on this platform)");
        return false;
    }
    true
}

/// Open an isolated bucket on the lazy shared session (blocking).
pub fn open_test_fixture(prefix: &str) -> Result<RustFsTestFixture, String> {
    prefetch_rustfs_binary();
    let lease = session()?.open_test_bucket(prefix)?;
    let bucket = lease.bucket.clone();
    let storage = Arc::clone(&lease.storage);
    drop(lease);
    Ok(RustFsTestFixture { storage, bucket })
}

/// [`open_test_fixture`] on a blocking thread (for async tests).
pub async fn open_test_fixture_async(prefix: &str) -> Result<RustFsTestFixture, String> {
    let prefix = prefix.to_string();
    tokio::task::spawn_blocking(move || open_test_fixture(&prefix))
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
}

/// Drop a cleanup lease on a blocking thread (bucket empty + delete).
pub async fn release_lease(lease: RustFsBucketLease) {
    tokio::task::spawn_blocking(move || drop(lease))
        .await
        .expect("spawn_blocking join for lease release");
}

/// Isolated [`S3StorageProvider`] on the shared session for wire tests that need
/// the concrete type (same bucket path as [`open_test_fixture`]).
pub fn open_wire_test_provider() -> Result<S3StorageProvider, String> {
    let fixture = open_test_fixture("")?;
    let session = session()?;
    S3StorageProvider::new_with_endpoint_and_prefix(
        session.endpoint(),
        &fixture.bucket,
        session.access_key(),
        session.secret_key(),
        RUSTFS_REGION,
        "",
        Some(session.ca_pem()),
    )
    .map_err(|e| e.to_string())
}

fn keep_rustfs_bucket() -> bool {
    std::env::var_os("INFINO_BENCH_KEEP_TABLE").is_some()
}

fn unique_rustfs_bucket_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = RUSTFS_BUCKET_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("infino-{}-{nanos}-{seq}", std::process::id())
}

fn provision_bucket(
    endpoint: &str,
    access_key: &str,
    secret_key: &str,
    ca_pem: &[u8],
    bucket: &str,
) -> Result<(), String> {
    create_bucket(
        endpoint,
        bucket,
        access_key,
        secret_key,
        RUSTFS_REGION,
        ca_pem,
    )
}

fn empty_and_delete_bucket(session: &RustFsSession, bucket: &str) -> Result<(), String> {
    for _ in 0..BUCKET_EMPTY_MAX_PASSES {
        let keys = list_bucket_object_keys(session, bucket)?;
        if keys.is_empty() {
            break;
        }
        for key in keys {
            delete_object(session, bucket, &key)?;
        }
    }
    let remaining = list_bucket_object_keys(session, bucket)?;
    if !remaining.is_empty() {
        return Err(format!(
            "bucket {bucket} still has {} object(s) after {BUCKET_EMPTY_MAX_PASSES} empty passes",
            remaining.len()
        ));
    }
    delete_bucket(
        &session.endpoint,
        bucket,
        &session.access_key,
        &session.secret_key,
        RUSTFS_REGION,
        &session.ca_pem,
    )
}

/// List every object key in `bucket` via blocking SigV4 GET (ListObjectsV2).
fn list_bucket_object_keys(session: &RustFsSession, bucket: &str) -> Result<Vec<String>, String> {
    let mut keys = Vec::new();
    let mut continuation: Option<String> = None;
    loop {
        let (page, next, truncated) =
            list_bucket_object_keys_page(session, bucket, continuation.as_deref())?;
        keys.extend(page);
        if !truncated {
            break;
        }
        let next = next.ok_or_else(|| {
            format!(
                "ListObjectsV2 for {bucket} returned truncated=true without NextContinuationToken"
            )
        })?;
        if continuation.as_deref() == Some(next.as_str()) {
            return Err(format!(
                "ListObjectsV2 for {bucket} repeated continuation token"
            ));
        }
        continuation = Some(next);
    }
    Ok(keys)
}

fn list_bucket_object_keys_page(
    session: &RustFsSession,
    bucket: &str,
    continuation_token: Option<&str>,
) -> Result<(Vec<String>, Option<String>, bool), String> {
    let mut query_pairs = vec![("list-type", "2")];
    if let Some(token) = continuation_token {
        query_pairs.push(("continuation-token", token));
    }
    let canonical_query = canonical_query_string(&query_pairs);
    let query_suffix = if canonical_query.is_empty() {
        String::new()
    } else {
        format!("?{canonical_query}")
    };
    let body = signed_s3_get(
        session,
        bucket,
        &format!("/{bucket}{query_suffix}"),
        &canonical_query,
    )?;
    parse_list_objects_v2_page(&body)
}

fn delete_object(session: &RustFsSession, bucket: &str, key: &str) -> Result<(), String> {
    let encoded_key = percent_encode_path_segment(key);
    let canonical_uri = format!("/{bucket}/{encoded_key}");
    signed_s3_delete(session, &canonical_uri)
}

fn rustfs_blocking_client(ca_pem: &[u8]) -> Result<reqwest::blocking::Client, String> {
    let cert = reqwest::Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    reqwest::blocking::Client::builder()
        .add_root_certificate(cert)
        .build()
        .map_err(|e| e.to_string())
}

fn canonical_uri_from_request_target(request_target: &str) -> &str {
    request_target
        .split_once('?')
        .map_or(request_target, |(path, _)| path)
}

fn signed_s3_get(
    session: &RustFsSession,
    bucket: &str,
    request_target: &str,
    canonical_query: &str,
) -> Result<String, String> {
    let client = rustfs_blocking_client(&session.ca_pem)?;
    let host = host_header(&session.endpoint)?;
    let url = format!("{}{request_target}", session.endpoint);
    let amz_date = amz_timestamp();
    let payload_hash = sha256_hex(b"");
    let canonical_uri = canonical_uri_from_request_target(request_target);
    let authorization = sign_s3_request(&S3SignParams {
        method: "GET",
        canonical_uri,
        canonical_query,
        host: &host,
        amz_date: &amz_date,
        payload_hash: &payload_hash,
        access_key: &session.access_key,
        secret_key: &session.secret_key,
        region: RUSTFS_REGION,
    })?;
    let response = client
        .get(&url)
        .header("host", &host)
        .header("x-amz-date", &amz_date)
        .header("x-amz-content-sha256", &payload_hash)
        .header("authorization", authorization)
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    let body = response.text().map_err(|e| e.to_string())?;
    if status.is_success() {
        return Ok(body);
    }
    Err(format!(
        "ListObjectsV2 failed for {bucket}: HTTP {status} {body}"
    ))
}

fn signed_s3_delete(session: &RustFsSession, canonical_uri: &str) -> Result<(), String> {
    let client = rustfs_blocking_client(&session.ca_pem)?;
    let host = host_header(&session.endpoint)?;
    let url = format!("{}{canonical_uri}", session.endpoint);
    let amz_date = amz_timestamp();
    let payload_hash = sha256_hex(b"");
    let authorization = sign_s3_request(&S3SignParams {
        method: "DELETE",
        canonical_uri,
        canonical_query: "",
        host: &host,
        amz_date: &amz_date,
        payload_hash: &payload_hash,
        access_key: &session.access_key,
        secret_key: &session.secret_key,
        region: RUSTFS_REGION,
    })?;
    let response = client
        .delete(&url)
        .header("host", &host)
        .header("x-amz-date", &amz_date)
        .header("x-amz-content-sha256", &payload_hash)
        .header("authorization", authorization)
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    let body = response.text().ok();
    if status.is_success() || status.as_u16() == 404 {
        return Ok(());
    }
    Err(format!(
        "DeleteObject failed for {canonical_uri}: HTTP {status} {body:?}"
    ))
}

fn canonical_query_string(pairs: &[(&str, &str)]) -> String {
    let mut encoded: Vec<String> = pairs
        .iter()
        .map(|(k, v)| {
            format!(
                "{}={}",
                percent_encode_query_component(k),
                percent_encode_query_component(v)
            )
        })
        .collect();
    encoded.sort();
    encoded.join("&")
}

fn percent_encode_query_component(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn percent_encode_path_segment(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Decode XML character references in S3 `ListObjectsV2` text nodes.
fn unescape_xml_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        if let Some((decoded, consumed)) = decode_xml_entity(rest) {
            out.push(decoded);
            rest = &rest[consumed..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

fn decode_xml_entity(s: &str) -> Option<(char, usize)> {
    if !s.starts_with('&') {
        return None;
    }
    let end = s.find(';')?;
    let entity = &s[1..end];
    let consumed = end + 1;
    if let Some(hex) = entity.strip_prefix("#x") {
        let code = u32::from_str_radix(hex, 16).ok()?;
        return char::from_u32(code).map(|ch| (ch, consumed));
    }
    if let Some(decimal) = entity.strip_prefix('#') {
        let code = decimal.parse::<u32>().ok()?;
        return char::from_u32(code).map(|ch| (ch, consumed));
    }
    let ch = match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        _ => return None,
    };
    Some((ch, consumed))
}

fn parse_list_objects_v2_page(xml: &str) -> Result<(Vec<String>, Option<String>, bool), String> {
    let mut keys = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Key>") {
        rest = &rest[start + "<Key>".len()..];
        let end = rest
            .find("</Key>")
            .ok_or("ListObjectsV2 response missing </Key>")?;
        keys.push(unescape_xml_entities(&rest[..end]));
        rest = &rest[end..];
    }
    let truncated = xml.contains("<IsTruncated>true</IsTruncated>");
    let continuation = xml
        .split("<NextContinuationToken>")
        .nth(1)
        .and_then(|tail| tail.split("</NextContinuationToken>").next())
        .map(unescape_xml_entities);
    Ok((keys, continuation, truncated))
}

/// Build an HTTPS S3 provider that trusts the bench-local CA.
pub fn rustfs_s3_provider(
    handle: &RustFsHandle,
    prefix: &str,
) -> Result<Arc<dyn StorageProvider>, String> {
    build_rustfs_provider(
        &handle.endpoint,
        &handle.bucket,
        prefix,
        &handle.access_key,
        &handle.secret_key,
        &handle.ca_pem,
    )
}

fn build_rustfs_provider(
    endpoint: &str,
    bucket: &str,
    prefix: &str,
    access_key: &str,
    secret_key: &str,
    ca_pem: &[u8],
) -> Result<Arc<dyn StorageProvider>, String> {
    let provider = S3StorageProvider::new_with_endpoint_and_prefix(
        endpoint,
        bucket,
        access_key,
        secret_key,
        RUSTFS_REGION,
        prefix,
        Some(ca_pem),
    )
    .map_err(|e| e.to_string())?;
    Ok(Arc::new(provider))
}

fn rustfs_credentials() -> (String, String) {
    let access_key =
        std::env::var("RUSTFS_ACCESS_KEY").unwrap_or_else(|_| generate_key(GENERATED_KEY_LEN));
    let secret_key =
        std::env::var("RUSTFS_SECRET_KEY").unwrap_or_else(|_| generate_key(GENERATED_SECRET_LEN));
    (access_key, secret_key)
}

fn generate_key(len: usize) -> String {
    let mut bytes = vec![0_u8; len];
    rand::fill(&mut bytes);
    bytes
        .into_iter()
        .map(|b| {
            let idx = (b as usize) % GENERATED_KEY_CHARSET.len();
            GENERATED_KEY_CHARSET[idx] as char
        })
        .collect()
}

fn rustfs_cache_dir() -> PathBuf {
    std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"))
        .join("infino-bench")
        .join("rustfs")
}

fn rustfs_cache_binary_path() -> PathBuf {
    rustfs_cache_dir().join("rustfs")
}

fn rustfs_cache_version_path() -> PathBuf {
    rustfs_cache_dir().join("rustfs.version")
}

fn read_rustfs_cache_version() -> Option<String> {
    std::fs::read_to_string(rustfs_cache_version_path())
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|version| !version.is_empty())
}

fn write_rustfs_cache_version(version: &str) -> Result<(), String> {
    std::fs::create_dir_all(rustfs_cache_dir()).map_err(|e| e.to_string())?;
    std::fs::write(rustfs_cache_version_path(), format!("{version}\n")).map_err(|e| e.to_string())
}

/// True when `rustfs.version` was written recently enough to skip a version-string
/// compare (and any re-download) until the recheck window elapses.
fn rustfs_version_stamp_within_recheck_window() -> bool {
    let path = rustfs_cache_version_path();
    let Ok(meta) = std::fs::metadata(&path) else {
        return false;
    };
    let stamp = version_stamp_system_time(&meta);
    let Ok(age) = SystemTime::now().duration_since(stamp) else {
        return false;
    };
    age <= Duration::from_secs(RUSTFS_CACHE_VERSION_RECHECK_SECS)
}

#[cfg(unix)]
fn version_stamp_system_time(meta: &std::fs::Metadata) -> SystemTime {
    use std::os::unix::fs::MetadataExt;
    SystemTime::UNIX_EPOCH + Duration::from_secs(meta.ctime() as u64)
}

#[cfg(not(unix))]
fn version_stamp_system_time(meta: &std::fs::Metadata) -> SystemTime {
    meta.modified().unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Drop the auto-downloaded binary and its version stamp (not `INFINO_RUSTFS_BIN` / PATH).
fn clear_rustfs_cache() {
    let _ = std::fs::remove_file(rustfs_cache_binary_path());
    let _ = std::fs::remove_file(rustfs_cache_version_path());
    let _ = std::fs::remove_file(rustfs_cache_binary_path().with_extension("tmp"));
}

fn rustfs_version() -> String {
    std::env::var("INFINO_RUSTFS_VERSION").unwrap_or_else(|_| DEFAULT_RUSTFS_VERSION.into())
}

fn which_rustfs_on_path() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            #[cfg(windows)]
            {
                let exe = dir.join("rustfs.exe");
                if exe.is_file() {
                    return Some(exe);
                }
            }
            let candidate = dir.join("rustfs");
            candidate.is_file().then_some(candidate)
        })
    })
}

/// Progress for long RustFS setup (binary download). Uses `/dev/tty` when
/// available so lines are visible during `cargo test`, which captures stderr.
fn log_rustfs_progress(line: &str) {
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        if let Ok(mut tty) = OpenOptions::new().write(true).open("/dev/tty") {
            let _ = writeln!(tty, "{line}");
            return;
        }
    }
    eprintln!("{line}");
}

fn download_rustfs_binary(dest: &Path) -> Result<(), String> {
    let version = rustfs_version();
    let asset = release_asset_name(&version)?;
    let release_base = format!("https://github.com/rustfs/rustfs/releases/download/{version}");
    let url = format!("{release_base}/{asset}");

    log_rustfs_progress("[rustfs] cold start: downloading RustFS binary for tests and benches");
    log_rustfs_progress("(To disable RustFS S3 tests: set INFINO_TEST_DISABLE_RUSTFS=1)");
    log_rustfs_progress(&format!("[rustfs]   URL: {url}"));
    match github_head_content_length(&url) {
        Some(len) => log_rustfs_progress(&format!(
            "[rustfs]   size: {} (Content-Length)",
            fmt_bytes(len)
        )),
        None => log_rustfs_progress(
            "[rustfs]   size: unknown (Content-Length unavailable; download may take a few minutes)",
        ),
    }

    std::fs::create_dir_all(
        dest.parent()
            .ok_or_else(|| "rustfs cache path has no parent".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let zip_bytes = github_bytes(&url)?;
    log_rustfs_progress(&format!(
        "[rustfs]   downloaded {} zip archive",
        fmt_bytes(zip_bytes.len() as u64)
    ));
    verify_release_sha256(&zip_bytes, &release_base, &asset)?;

    let reader = Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| e.to_string())?;
    let tmp = dest.with_extension("tmp");
    let mut extracted = false;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        if name.ends_with("rustfs") || name.ends_with("rustfs.exe") {
            let mut out = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut out).map_err(|e| e.to_string())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = out.metadata().map_err(|e| e.to_string())?.permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(&tmp, perms).map_err(|e| e.to_string())?;
            }
            std::fs::rename(&tmp, dest).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                e.to_string()
            })?;
            extracted = true;
            break;
        }
    }
    if !extracted {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("rustfs binary not found inside {asset}"));
    }
    let installed_len = std::fs::metadata(dest).map(|meta| meta.len()).unwrap_or(0);
    log_rustfs_progress(&format!(
        "[rustfs] installed binary at {} ({})",
        dest.display(),
        fmt_bytes(installed_len)
    ));
    write_rustfs_cache_version(&version)?;
    Ok(())
}

fn github_http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .build()
        .map_err(|e| e.to_string())
}

/// `Content-Length` from a release asset URL, when GitHub exposes it on HEAD.
fn github_head_content_length(url: &str) -> Option<u64> {
    let client = github_http_client().ok()?;
    let response = client.head(url).send().ok()?.error_for_status().ok()?;
    response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|text| text.parse().ok())
}

/// Fetch a public GitHub release asset over HTTPS (system trust roots).
///
/// Follows redirects (3xx). Only 2xx responses return a body; 4xx/5xx fail fast.
fn github_bytes(url: &str) -> Result<Vec<u8>, String> {
    let client = github_http_client()?;
    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("GET {url} failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("GET {url} failed: {e}"))?;
    Ok(response.bytes().map_err(|e| e.to_string())?.to_vec())
}

fn verify_release_sha256(zip_bytes: &[u8], release_base: &str, asset: &str) -> Result<(), String> {
    let sums_url = format!("{release_base}/{RUSTFS_SHA256SUMS_ASSET}");
    log_rustfs_progress(&format!(
        "[rustfs] verifying {asset} against {RUSTFS_SHA256SUMS_ASSET} ..."
    ));
    let sums_text = String::from_utf8(github_bytes(&sums_url)?)
        .map_err(|e| format!("{RUSTFS_SHA256SUMS_ASSET} is not valid UTF-8: {e}"))?;
    let expected = parse_sha256_sums_entry(&sums_text, asset)?;
    let actual = sha256_hex(zip_bytes);
    if actual != expected {
        return Err(format!(
            "rustfs {asset} SHA256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn parse_sha256_sums_entry(sums: &str, asset: &str) -> Result<String, String> {
    for line in sums.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(hash) = parts.next() else {
            continue;
        };
        let Some(name) = parts.next() else {
            continue;
        };
        let name = name.strip_prefix('*').unwrap_or(name);
        if name == asset {
            return Ok(hash.to_ascii_lowercase());
        }
    }
    Err(format!(
        "{RUSTFS_SHA256SUMS_ASSET} has no entry for {asset}"
    ))
}

fn child_exited(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(Some(_)))
}

fn release_asset_name(version: &str) -> Result<String, String> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    // Upstream release zips use a `v` prefix on the tag in the filename
    // (e.g. `rustfs-linux-x86_64-gnu-v1.0.0-alpha.90.zip`) while the
    // GitHub release tag path omits it (`.../download/1.0.0-alpha.90/`).
    let versioned = format!("v{version}");
    let stem = match (os, arch) {
        ("linux", "x86_64") => format!("rustfs-linux-x86_64-gnu-{versioned}.zip"),
        ("linux", "aarch64") => format!("rustfs-linux-aarch64-gnu-{versioned}.zip"),
        ("macos", "x86_64") => format!("rustfs-macos-x86_64-{versioned}.zip"),
        ("macos", "aarch64") => format!("rustfs-macos-aarch64-{versioned}.zip"),
        ("windows", "x86_64") => format!("rustfs-windows-x86_64-{versioned}.zip"),
        _ => {
            return Err(format!(
                "unsupported platform for auto-download: {os}-{arch}"
            ));
        }
    };
    Ok(stem)
}

/// Pick a random loopback port without binding in this process.
///
/// RustFS binds the port itself, so we cannot hold a listener open here.
/// Pre-binding and dropping would also introduce a TOCTOU race. Instead we
/// choose from the ephemeral range and rely on [`spawn_rustfs`] retries when
/// the port is already taken.
fn random_loopback_addr() -> SocketAddr {
    use rand::RngExt;
    let port = rand::rng().random_range(EPHEMERAL_PORT_MIN..=EPHEMERAL_PORT_MAX);
    SocketAddr::from(([127, 0, 0, 1], port))
}

fn generate_tls_material() -> Result<(TempDir, Vec<u8>), String> {
    let tls_dir = TempDir::new().map_err(|e| e.to_string())?;

    let ca_key = KeyPair::generate().map_err(|e| e.to_string())?;
    let mut ca_params = CertificateParams::default();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params
        .distinguished_name
        .push(DnType::CommonName, "Infino Test CA");
    let ca_cert = ca_params.self_signed(&ca_key).map_err(|e| e.to_string())?;
    let ca_pem = ca_cert.pem().into_bytes();

    let server_key = KeyPair::generate().map_err(|e| e.to_string())?;
    let mut server_params =
        CertificateParams::new(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .map_err(|e| e.to_string())?;
    server_params.serial_number = Some(SerialNumber::from(1_u64));
    server_params
        .distinguished_name
        .push(DnType::CommonName, "localhost");
    let issuer = Issuer::from_params(&ca_params, &ca_key);
    let server_cert = server_params
        .signed_by(&server_key, &issuer)
        .map_err(|e| e.to_string())?;

    let cert_path = tls_dir.path().join("rustfs_cert.pem");
    let key_path = tls_dir.path().join("rustfs_key.pem");
    std::fs::write(&cert_path, server_cert.pem()).map_err(|e| e.to_string())?;
    std::fs::write(&key_path, server_key.serialize_pem()).map_err(|e| e.to_string())?;

    Ok((tls_dir, ca_pem))
}

fn wait_for_health(child: &mut Child, endpoint: &str, ca_pem: &[u8]) -> Result<(), String> {
    let url = format!("{endpoint}/health");
    let cert = reqwest::Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    let client = reqwest::blocking::Client::builder()
        .add_root_certificate(cert)
        .build()
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    while Instant::now() < deadline {
        if child_exited(child) {
            return Err(format!(
                "rustfs exited before health check succeeded at {url}"
            ));
        }
        if let Ok(response) = client.get(&url).send()
            && response.status().is_success()
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(HEALTH_POLL_INTERVAL_MS));
    }
    Err(format!("rustfs health check timed out at {url}"))
}

fn create_bucket(
    endpoint: &str,
    bucket: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    ca_pem: &[u8],
) -> Result<(), String> {
    let cert = reqwest::Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    let client = reqwest::blocking::Client::builder()
        .add_root_certificate(cert)
        .build()
        .map_err(|e| e.to_string())?;
    let host = host_header(endpoint)?;
    let url = format!("{endpoint}/{bucket}");
    let amz_date = amz_timestamp();
    let payload_hash = sha256_hex(b"");
    let authorization = sign_s3_request(&S3SignParams {
        method: "PUT",
        canonical_uri: &format!("/{bucket}"),
        canonical_query: "",
        host: &host,
        amz_date: &amz_date,
        payload_hash: &payload_hash,
        access_key,
        secret_key,
        region,
    })?;
    let response = client
        .put(&url)
        .header("host", &host)
        .header("x-amz-date", &amz_date)
        .header("x-amz-content-sha256", &payload_hash)
        .header("authorization", authorization)
        .body(Vec::<u8>::new())
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if status.is_success() || status.as_u16() == 409 {
        return Ok(());
    }
    Err(format!(
        "CreateBucket failed for {bucket}: HTTP {} {:?}",
        status,
        response.text().ok()
    ))
}

fn delete_bucket(
    endpoint: &str,
    bucket: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    ca_pem: &[u8],
) -> Result<(), String> {
    let cert = reqwest::Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    let client = reqwest::blocking::Client::builder()
        .add_root_certificate(cert)
        .build()
        .map_err(|e| e.to_string())?;
    let host = host_header(endpoint)?;
    let url = format!("{endpoint}/{bucket}");
    let amz_date = amz_timestamp();
    let payload_hash = sha256_hex(b"");
    let authorization = sign_s3_request(&S3SignParams {
        method: "DELETE",
        canonical_uri: &format!("/{bucket}"),
        canonical_query: "",
        host: &host,
        amz_date: &amz_date,
        payload_hash: &payload_hash,
        access_key,
        secret_key,
        region,
    })?;
    let response = client
        .delete(&url)
        .header("host", &host)
        .header("x-amz-date", &amz_date)
        .header("x-amz-content-sha256", &payload_hash)
        .header("authorization", authorization)
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if status.is_success() || status.as_u16() == 404 {
        return Ok(());
    }
    Err(format!(
        "DeleteBucket failed for {bucket}: HTTP {} {:?}",
        status,
        response.text().ok()
    ))
}

fn amz_timestamp() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string()
}

fn host_header(endpoint: &str) -> Result<String, String> {
    endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .map(str::to_string)
        .ok_or_else(|| format!("invalid rustfs endpoint: {endpoint}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

fn sign_s3_request(params: &S3SignParams<'_>) -> Result<String, String> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    let date_stamp = &params.amz_date[..8];
    let service = "s3";
    let credential_scope = format!("{date_stamp}/{}/{service}/aws4_request", params.region);
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";
    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        params.host, params.payload_hash, params.amz_date
    );
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{signed_headers}\n{}",
        params.method,
        params.canonical_uri,
        params.canonical_query,
        canonical_headers,
        params.payload_hash
    );
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{credential_scope}\n{}",
        params.amz_date,
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = HmacSha256::new_from_slice(format!("AWS4{}", params.secret_key).as_bytes())
        .map_err(|e| e.to_string())?
        .chain_update(date_stamp.as_bytes())
        .finalize()
        .into_bytes();
    let k_region = HmacSha256::new_from_slice(&k_date)
        .map_err(|e| e.to_string())?
        .chain_update(params.region.as_bytes())
        .finalize()
        .into_bytes();
    let k_service = HmacSha256::new_from_slice(&k_region)
        .map_err(|e| e.to_string())?
        .chain_update(service.as_bytes())
        .finalize()
        .into_bytes();
    let k_signing = HmacSha256::new_from_slice(&k_service)
        .map_err(|e| e.to_string())?
        .chain_update(b"aws4_request")
        .finalize()
        .into_bytes();
    let signature = hex::encode(
        HmacSha256::new_from_slice(&k_signing)
            .map_err(|e| e.to_string())?
            .chain_update(string_to_sign.as_bytes())
            .finalize()
            .into_bytes(),
    );

    Ok(format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
        params.access_key
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescape_xml_entities_decodes_numeric_references() {
        assert_eq!(unescape_xml_entities("a&#38;b"), "a&b");
        assert_eq!(unescape_xml_entities("a&#x26;b"), "a&b");
        assert_eq!(unescape_xml_entities("&#65;"), "A");
    }

    #[test]
    fn parse_list_objects_v2_page_unescapes_xml_entities() {
        let xml = "\
<ListBucketResult>
<Contents><Key>a&amp;b</Key></Contents>
<NextContinuationToken>tok&amp;1</NextContinuationToken>
</ListBucketResult>";
        let (keys, next, truncated) = parse_list_objects_v2_page(xml).expect("parse");
        assert_eq!(keys, vec!["a&b".to_string()]);
        assert_eq!(next.as_deref(), Some("tok&1"));
        assert!(!truncated);
    }

    #[test]
    fn parse_list_objects_v2_page_extracts_keys_and_continuation() {
        let xml = "\
<ListBucketResult>
<IsTruncated>true</IsTruncated>
<Contents><Key>a</Key></Contents>
<Contents><Key>b/c</Key></Contents>
<NextContinuationToken>tok</NextContinuationToken>
</ListBucketResult>";
        let (keys, next, truncated) = parse_list_objects_v2_page(xml).expect("parse");
        assert_eq!(keys, vec!["a".to_string(), "b/c".to_string()]);
        assert_eq!(next.as_deref(), Some("tok"));
        assert!(truncated);
    }

    #[test]
    fn parse_sha256_sums_finds_asset_line() {
        let sums = "\
abc123  rustfs-linux-x86_64-gnu-latest.zip
def456  other.zip
";
        assert_eq!(
            parse_sha256_sums_entry(sums, "rustfs-linux-x86_64-gnu-latest.zip").expect("parse"),
            "abc123"
        );
    }

    #[test]
    fn parse_sha256_sums_accepts_bsd_marker() {
        let sums = "deadbeef  *rustfs-macos-aarch64-latest.zip\n";
        assert_eq!(
            parse_sha256_sums_entry(sums, "rustfs-macos-aarch64-latest.zip").expect("parse"),
            "deadbeef"
        );
    }

    #[test]
    fn parse_sha256_sums_missing_asset_errors() {
        assert!(parse_sha256_sums_entry("abc123  other.zip\n", "missing.zip").is_err());
    }

    #[test]
    fn unique_rustfs_bucket_names_are_distinct_within_process() {
        let a = unique_rustfs_bucket_name();
        let b = unique_rustfs_bucket_name();
        assert_ne!(a, b, "bucket names must be unique within a process");
    }

    #[test]
    fn random_loopback_addr_is_ephemeral_loopback() {
        let addr = random_loopback_addr();
        assert_eq!(
            addr.ip(),
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
        assert!(
            (EPHEMERAL_PORT_MIN..=EPHEMERAL_PORT_MAX).contains(&addr.port()),
            "port {} outside ephemeral range",
            addr.port()
        );
    }

    #[test]
    fn release_asset_name_uses_v_prefix_in_zip_filename() {
        let version = "1.0.0-alpha.90";
        let expected = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("linux", "x86_64") => "rustfs-linux-x86_64-gnu-v1.0.0-alpha.90.zip",
            ("linux", "aarch64") => "rustfs-linux-aarch64-gnu-v1.0.0-alpha.90.zip",
            ("macos", "x86_64") => "rustfs-macos-x86_64-v1.0.0-alpha.90.zip",
            ("macos", "aarch64") => "rustfs-macos-aarch64-v1.0.0-alpha.90.zip",
            ("windows", "x86_64") => "rustfs-windows-x86_64-v1.0.0-alpha.90.zip",
            (os, arch) => panic!("unsupported platform for test: {os}-{arch}"),
        };
        assert_eq!(release_asset_name(version).expect("asset name"), expected);
    }
}
