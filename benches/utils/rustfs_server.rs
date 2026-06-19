// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Local RustFS daemon lifecycle for benchmark object-store fixtures.
//!
//! Spawns a child `rustfs` process with bench-generated TLS credentials,
//! waits for `/health`, creates the target bucket, and builds an HTTPS
//! `S3StorageProvider` that trusts the bench-local CA.

use std::{
    io::Cursor,
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use object_store::{
    Certificate, ClientOptions,
    aws::{AmazonS3Builder, S3ConditionalPut},
};
use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair, SerialNumber};
use tempfile::TempDir;

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
/// Spawn attempts when the reserved loopback port is taken before RustFS binds.
const RUSTFS_SPAWN_MAX_ATTEMPTS: u32 = 5;
/// Filename of the upstream checksum manifest on RustFS GitHub releases.
const RUSTFS_SHA256SUMS_ASSET: &str = "SHA256SUMS";

struct S3SignParams<'a> {
    method: &'a str,
    canonical_uri: &'a str,
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

/// Send SIGTERM to a spawned RustFS child, then SIGKILL if needed.
pub fn terminate_child(child: &mut Child) {
    terminate_child_impl(child);
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

    let cached = rustfs_cache_binary_path();
    if cached.is_file() {
        return Ok(cached);
    }

    if let Some(path) = which_rustfs_on_path() {
        return Ok(path);
    }

    download_rustfs_binary(&cached)?;
    Ok(cached)
}

/// Spawn RustFS on a random loopback port with HTTPS enabled.
pub fn spawn_rustfs(bucket: &str) -> Result<RustFsHandle, String> {
    let binary = ensure_rustfs_binary()?;
    let (access_key, secret_key) = rustfs_credentials();
    let data_dir = TempDir::new().map_err(|e| e.to_string())?;
    let (tls_dir, ca_pem) = generate_tls_material()?;

    let mut last_err = String::new();
    for attempt in 1..=RUSTFS_SPAWN_MAX_ATTEMPTS {
        let addr = reserve_loopback_port()?;
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

        match wait_for_health(&endpoint, &ca_pem) {
            Ok(()) => {
                create_bucket(
                    &endpoint,
                    bucket,
                    &access_key,
                    &secret_key,
                    RUSTFS_REGION,
                    &ca_pem,
                )?;
                eprintln!("[rustfs] endpoint={endpoint} bucket={bucket} storage_label=rustfs");
                return Ok(RustFsHandle {
                    endpoint,
                    bucket: bucket.to_string(),
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
    let cert = Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    let client_options = ClientOptions::new().with_root_certificate(cert);
    let store = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket)
        .with_access_key_id(access_key)
        .with_secret_access_key(secret_key)
        .with_region(RUSTFS_REGION)
        .with_virtual_hosted_style_request(false)
        .with_conditional_put(S3ConditionalPut::ETagMatch)
        .with_client_options(client_options)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(S3StorageProvider::from_object_store_with_prefix(
        bucket, store, prefix,
    )))
}

fn rustfs_credentials() -> (String, String) {
    let access_key =
        std::env::var("RUSTFS_ACCESS_KEY").unwrap_or_else(|_| generate_key(GENERATED_KEY_LEN));
    let secret_key =
        std::env::var("RUSTFS_SECRET_KEY").unwrap_or_else(|_| generate_key(GENERATED_SECRET_LEN));
    (access_key, secret_key)
}

fn generate_key(len: usize) -> String {
    use rand::RngExt;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    (0..len)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
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

fn rustfs_version() -> String {
    std::env::var("INFINO_RUSTFS_VERSION").unwrap_or_else(|_| DEFAULT_RUSTFS_VERSION.into())
}

fn which_rustfs_on_path() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("rustfs");
            candidate.is_file().then_some(candidate)
        })
    })
}

fn download_rustfs_binary(dest: &Path) -> Result<(), String> {
    let version = rustfs_version();
    let asset = release_asset_name()?;
    let release_base = format!("https://github.com/rustfs/rustfs/releases/download/{version}");
    let url = format!("{release_base}/{asset}");
    eprintln!("[rustfs] downloading {url} ...");

    std::fs::create_dir_all(
        dest.parent()
            .ok_or_else(|| "rustfs cache path has no parent".to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let zip_bytes = github_bytes(&url)?;
    verify_release_sha256(&zip_bytes, &release_base, &asset)?;

    let reader = Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| e.to_string())?;
    let mut extracted = false;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        if name.ends_with("rustfs") || name.ends_with("rustfs.exe") {
            let mut out = std::fs::File::create(dest).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut out).map_err(|e| e.to_string())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = out.metadata().map_err(|e| e.to_string())?.permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(dest, perms).map_err(|e| e.to_string())?;
            }
            extracted = true;
            break;
        }
    }
    if !extracted {
        return Err(format!("rustfs binary not found inside {asset}"));
    }
    eprintln!("[rustfs] installed binary at {}", dest.display());
    Ok(())
}

/// Fetch a public GitHub release asset over HTTPS (system trust roots).
///
/// Follows redirects (3xx). Only 2xx responses return a body; 4xx/5xx fail fast.
fn github_bytes(url: &str) -> Result<Vec<u8>, String> {
    let client = reqwest::blocking::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;
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
    eprintln!("[rustfs] verifying {asset} against {RUSTFS_SHA256SUMS_ASSET} ...");
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

fn release_asset_name() -> Result<String, String> {
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;
    let stem = match (os, arch) {
        ("linux", "x86_64") => "rustfs-linux-x86_64-gnu-latest.zip",
        ("linux", "aarch64") => "rustfs-linux-aarch64-gnu-latest.zip",
        ("macos", "x86_64") => "rustfs-macos-x86_64-latest.zip",
        ("macos", "aarch64") => "rustfs-macos-aarch64-latest.zip",
        ("windows", "x86_64") => "rustfs-windows-x86_64-latest.zip",
        _ => {
            return Err(format!(
                "unsupported platform for auto-download: {os}-{arch}"
            ));
        }
    };
    Ok(stem.to_string())
}

fn reserve_loopback_port() -> Result<SocketAddr, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    drop(listener);
    Ok(addr)
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

fn wait_for_health(endpoint: &str, ca_pem: &[u8]) -> Result<(), String> {
    let url = format!("{endpoint}/health");
    let cert = reqwest::Certificate::from_pem(ca_pem).map_err(|e| e.to_string())?;
    let client = reqwest::blocking::Client::builder()
        .add_root_certificate(cert)
        .build()
        .map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    while Instant::now() < deadline {
        if client.get(&url).send().is_ok() {
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
        "{}\n{}\n\n{}\n{signed_headers}\n{}",
        params.method, params.canonical_uri, canonical_headers, params.payload_hash
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
