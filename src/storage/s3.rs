// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! S3-backed [`StorageProvider`].
//!
//! Wraps `object_store::aws::AmazonS3` so the same supertable
//! code paths exercise both LocalFS (dev / tests / single-node
//! laptop scale) and S3 (production / multi-node) without
//! backend-specific branching above the storage trait.
//!
//! Compared to [`super::LocalFsStorageProvider`], the S3
//! variant uses native server-side conditional writes via S3
//! CAS (surfaced through `PutMode::Update(UpdateVersion)`).
//! There's no read-then-overwrite TOCTOU window on
//! `put_if_match`; the etag match is enforced atomically
//! server-side, returning `Error::Precondition` on conflict.
//!
//! ## Construction
//!
//! All credentials/region/endpoint come from a [`StorageOptions`] map
//! keyed by object_store's `aws_*` config strings — infino reads nothing
//! from the environment. [`Self::new_with_prefix`] is the primary path;
//! [`Self::new_with_endpoint`] is a convenience for RustFS / MinIO / Ceph.

use std::{ops::Range, str::FromStr, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{
    Certificate, ClientOptions, Error as ObjError, GetOptions, GetRange, MultipartUpload,
    ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion,
    aws::{AmazonS3, AmazonS3Builder, AmazonS3ConfigKey, S3ConditionalPut},
    path::Path as ObjPath,
};

use super::{
    ObjectMeta, StorageError, StorageOptions, StorageProvider, counting, io_counters,
    logical_list_key, options::apply, retry, usage::UsageMeter,
};

/// Whether `opts` names a custom S3 endpoint, under any object_store alias
/// (`aws_endpoint`, `endpoint`, `aws_endpoint_url`, …). A custom endpoint
/// selects the S3-compatible build profile (path-style, default client
/// options) over the AWS one.
fn has_custom_endpoint(opts: &StorageOptions) -> bool {
    opts.keys().any(|k| {
        matches!(
            AmazonS3ConfigKey::from_str(k),
            Ok(AmazonS3ConfigKey::Endpoint | AmazonS3ConfigKey::S3Endpoint)
        )
    })
}

/// S3-backed `StorageProvider`. Cheap to clone; the inner
/// `AmazonS3` shares its HTTP client across clones.
#[derive(Debug)]
pub struct S3StorageProvider {
    bucket: String,
    prefix: String,
    store: Arc<AmazonS3>,
    meter: Arc<UsageMeter>,
}

impl S3StorageProvider {
    /// S3 provider for `bucket` with no explicit options — credentials
    /// resolve through object_store's ambient chain (IAM role / workload
    /// identity). Infino never reads AWS credentials from the process
    /// environment; pass them through [`Self::new_with_prefix`] otherwise.
    pub fn new(bucket: impl Into<String>) -> Result<Self, StorageError> {
        Self::new_with_prefix(bucket, "", &StorageOptions::new())
    }

    /// S3 provider scoped to `prefix` inside `bucket`, configured from
    /// `opts` (credentials/region/endpoint, keyed by object_store's
    /// `aws_*` strings). A custom `aws_endpoint` switches to path-style +
    /// default client options; the tuned connection pool is AWS-only (it
    /// destabilizes local MinIO / RustFS endpoints).
    pub fn new_with_prefix(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        opts: &StorageOptions,
    ) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let uri = format!("s3://{bucket}");

        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(&bucket)
            .with_conditional_put(S3ConditionalPut::ETagMatch)
            .with_retry(retry::config());
        builder = if has_custom_endpoint(opts) {
            builder.with_virtual_hosted_style_request(false)
        } else {
            builder.with_client_options(tuned_client_options())
        };
        // Caller options last so they win (e.g. `aws_allow_http=true`).
        let builder = apply::<AmazonS3ConfigKey, _>(builder, opts, &uri, |b, key, value| {
            b.with_config(key, value)
        })?;

        let store = builder.build().map_err(|e| StorageError::Permanent {
            uri,
            source: Box::new(e),
        })?;
        Ok(Self {
            bucket,
            prefix: normalize_prefix(prefix),
            store: Arc::new(store),
            meter: UsageMeter::process_default(),
        })
    }

    /// Custom S3-compatible endpoint with static credentials (RustFS HTTPS,
    /// MinIO, Ceph).
    ///
    /// When `trusted_ca_pem` is `Some`, the PEM is installed as an
    /// additional root for TLS (local RustFS HTTPS). When `None`,
    /// `allow_http` is enabled so plain-HTTP endpoints are not
    /// rejected by the AWS SDK's HTTPS check.
    pub fn new_with_endpoint(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
        trusted_ca_pem: Option<&[u8]>,
    ) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let endpoint = endpoint.into();
        let store = build_custom_endpoint_store(
            &endpoint,
            &bucket,
            access_key,
            secret_key,
            region,
            trusted_ca_pem,
        )?;
        Ok(Self {
            bucket,
            prefix: String::new(),
            store: Arc::new(store),
            meter: UsageMeter::process_default(),
        })
    }

    /// Custom-endpoint variant of [`Self::new_with_prefix`] for
    /// S3-compatible deployments that also want a logical table prefix.
    pub fn new_with_endpoint_and_prefix(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
        prefix: impl Into<String>,
        trusted_ca_pem: Option<&[u8]>,
    ) -> Result<Self, StorageError> {
        let mut provider = Self::new_with_endpoint(
            endpoint,
            bucket,
            access_key,
            secret_key,
            region,
            trusted_ca_pem,
        )?;
        provider.prefix = normalize_prefix(prefix);
        Ok(provider)
    }

    /// Wrap an already-constructed `AmazonS3` — for advanced
    /// callers that want full control over the
    /// `AmazonS3Builder` (custom retry config, virtual-hosted
    /// vs path-style addressing, etc.).
    pub fn from_object_store(bucket: impl Into<String>, store: AmazonS3) -> Self {
        Self {
            bucket: bucket.into(),
            prefix: String::new(),
            store: Arc::new(store),
            meter: UsageMeter::process_default(),
        }
    }

    /// Replace the usage meter (connection-scoped ledger).
    pub fn with_usage_meter(mut self, meter: Arc<UsageMeter>) -> Self {
        self.meter = meter;
        self
    }

    /// [`Self::from_object_store`] with a logical table prefix,
    /// mirroring [`Self::new_with_prefix`].
    pub fn from_object_store_with_prefix(
        bucket: impl Into<String>,
        store: AmazonS3,
        prefix: impl Into<String>,
    ) -> Self {
        let mut provider = Self::from_object_store(bucket, store);
        provider.prefix = normalize_prefix(prefix);
        provider
    }

    /// S3 bucket this provider is scoped to.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Logical prefix prepended to every object path.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    fn key(&self, uri: &str) -> String {
        let uri = uri.trim_start_matches('/');
        if self.prefix.is_empty() {
            uri.to_string()
        } else {
            format!("{}/{uri}", self.prefix)
        }
    }

    fn path(&self, uri: &str) -> Result<ObjPath, StorageError> {
        let key = self.key(uri);
        ObjPath::parse(&key).map_err(|e| StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(e),
        })
    }
}

fn normalize_prefix(prefix: impl Into<String>) -> String {
    prefix.into().trim_matches('/').to_string()
}

/// Shared builder for [`S3StorageProvider::new_with_endpoint`].
fn build_custom_endpoint_store(
    endpoint: &str,
    bucket: &str,
    access_key: impl Into<String>,
    secret_key: impl Into<String>,
    region: impl Into<String>,
    trusted_ca_pem: Option<&[u8]>,
) -> Result<AmazonS3, StorageError> {
    let mut builder = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket)
        .with_access_key_id(access_key.into())
        .with_secret_access_key(secret_key.into())
        .with_region(region.into())
        // Force path-style addressing (bucket as path prefix, not subdomain).
        // Required for localhost-style endpoints (RustFS, MinIO, any
        // non-AWS S3-compatible service that doesn't terminate
        // `<bucket>.<endpoint>` DNS).
        .with_virtual_hosted_style_request(false)
        .with_conditional_put(S3ConditionalPut::ETagMatch);
    if let Some(ca_pem) = trusted_ca_pem {
        let cert = Certificate::from_pem(ca_pem).map_err(|e| StorageError::Permanent {
            uri: format!("s3://{bucket} @ {endpoint}"),
            source: Box::new(e),
        })?;
        let client_options = ClientOptions::new().with_root_certificate(cert);
        builder = builder.with_client_options(client_options);
    } else {
        // Plain-HTTP custom endpoints (MinIO on loopback, legacy emulators).
        // NB: do NOT apply `tuned_client_options()` here — the deep idle pool
        // destabilizes local S3-compatible endpoints. Also skip
        // `with_client_options` so this `with_allow_http` is not clobbered.
        builder = builder.with_allow_http(true);
    }
    builder.build().map_err(|e| StorageError::Permanent {
        uri: format!("s3://{bucket} @ {endpoint}"),
        source: Box::new(e),
    })
}

/// Warm idle connections kept per host. A deep pool lets a wide
/// concurrent range-GET fan-out reuse established TLS sessions
/// rather than re-handshaking on the cold tail.
const S3_POOL_MAX_IDLE_PER_HOST: usize = 1024;

/// Client idle-connection timeout. Held below S3's ~20s server-side
/// idle-close window so reqwest never reuses a socket S3 has already
/// dropped (which surfaces as a transient send failure).
const S3_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

/// Connect-phase timeout. Bounds a single slow SYN/TLS so it can't
/// dominate the fan-out's p99; the retry layer covers genuine drops.
const S3_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Tuned HTTP client options for the object-store-native fan-out.
///
/// The supertable vector/FTS query path fans out one cold-open +
/// cold-search batch per superfile concurrently. With the default
/// idle-connection pool, a wide fan-out (hundreds of superfiles ×
/// several range GETs each) churns TCP/TLS connections — each new
/// connection pays a TLS handshake RTT on top of the request RTT,
/// inflating the p99 tail under load. Keeping a large warm idle
/// pool lets the fan-out reuse connections so the per-GET cost is
/// one RTT, not handshake + RTT.
fn tuned_client_options() -> ClientOptions {
    ClientOptions::new()
        // Keep many connections warm per host so concurrent
        // fan-out GETs reuse established TLS sessions instead of
        // handshaking. AWS S3 in-region serves many parallel
        // range GETs per host; a deep idle pool is the difference
        // between "RTT" and "handshake + RTT" on the cold tail.
        .with_pool_max_idle_per_host(S3_POOL_MAX_IDLE_PER_HOST)
        // Hold idle connections long enough to span a full fan-out
        // wave plus the next query so back-to-back cold queries on a
        // fresh worker don't re-handshake — but keep this *below* S3's
        // server-side idle-close window. AWS closes idle keep-alive
        // connections after ~20s; a longer client idle timeout means
        // reqwest pools sockets S3 has already dropped, then reuses
        // one on the next bursty fan-out and fails the send with
        // "error sending request" (object_store retries, then
        // surfaces `TransientExhausted`). 10s keeps the pool warm
        // across consecutive queries while expiring sockets before
        // S3 can close them under us.
        .with_pool_idle_timeout(S3_POOL_IDLE_TIMEOUT)
        // Bound the connect phase so a single slow SYN/TLS doesn't
        // dominate the fan-out's p99; the retry layer covers drops.
        .with_connect_timeout(S3_CONNECT_TIMEOUT)
}

/// Translate an `object_store::Error` to our `StorageError`.
/// Same shape as the LocalFS provider's translate; kept here
/// rather than shared to keep each backend file self-
/// contained (the error mappings may diverge if S3's surface
/// of errors widens).
fn translate(uri: &str, e: ObjError) -> StorageError {
    match e {
        ObjError::NotFound { .. } => StorageError::NotFound { uri: uri.into() },
        ObjError::AlreadyExists { .. } | ObjError::Precondition { .. } => {
            StorageError::PreconditionFailed { uri: uri.into() }
        }
        ObjError::Generic { source, .. } => StorageError::TransientExhausted {
            uri: uri.into(),
            source,
        },
        other => StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(other),
        },
    }
}

#[async_trait]
impl StorageProvider for S3StorageProvider {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        let path = self.path(uri)?;
        let meta = self
            .store
            .head(&path)
            .await
            .map_err(|e| translate(uri, e))?;
        self.meter.record_head();
        Ok(ObjectMeta {
            size: meta.size as u64,
            etag: meta.e_tag,
            last_modified: meta.last_modified.into(),
        })
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = self.path(uri)?;
        let tl = io_counters::timeline_start();
        // etag and bytes are atomically paired in the same response, so
        // no follow-up HEAD is needed.
        let out = retry::complete_get(uri, || async {
            let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
            let meta = ObjectMeta {
                size: result.meta.size as u64,
                etag: result.meta.e_tag.clone(),
                last_modified: result.meta.last_modified.into(),
            };
            let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
            Ok((bytes, meta))
        })
        .await;
        if let Ok((b, _)) = &out {
            self.meter.record_get(uri, None, b.len() as u64);
            io_counters::timeline_record("get", uri, 0, b.len() as u64, tl);
        }
        out
    }

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        let path = self.path(uri)?;
        // Native `If-None-Match`: an unchanged object comes back as a
        // bodyless 304 instead of a full read. Both arms are still a
        // billable GET (0 body bytes on 304).
        let out = retry::with_reissue(|| async {
            let options = GetOptions {
                if_none_match: Some(etag.to_string()),
                ..GetOptions::default()
            };
            let result = match self.store.get_opts(&path, options).await {
                Ok(result) => result,
                Err(ObjError::NotModified { .. }) => return Ok(None),
                Err(e) => return Err(translate(uri, e)),
            };
            let meta = ObjectMeta {
                size: result.meta.size as u64,
                etag: result.meta.e_tag.clone(),
                last_modified: result.meta.last_modified.into(),
            };
            let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
            Ok(Some((bytes, meta)))
        })
        .await;
        match &out {
            Ok(Some((b, _))) => self.meter.record_get(uri, None, b.len() as u64),
            Ok(None) => self.meter.record_get(uri, None, 0),
            Err(_) => {}
        }
        out
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(uri = uri, len = range.end - range.start))
    )]
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = self.path(uri)?;
        let requested = (range.start, range.end);
        let off = range.start;
        let tl = io_counters::timeline_start();
        let out = retry::complete_range(uri, range, |r| async {
            self.store
                .get_range(&path, r)
                .await
                .map_err(|e| translate(uri, e))
        })
        .await;
        if let Ok(b) = &out {
            self.meter.record_get(uri, Some(requested), b.len() as u64);
            io_counters::timeline_record("get_range", uri, off, b.len() as u64, tl);
        }
        out
    }

    /// Tail-fetch path: — single-RTT tail fetch via S3's native
    /// `Range: bytes=-len` suffix-range form. The response
    /// carries the total object size in `GetResult::meta.size`,
    /// so callers don't need a separate HEAD round-trip just
    /// to learn the size.
    ///
    /// Compared to the default trait impl (HEAD + bounded
    /// GET = 2 RTTs), this collapses to 1 RTT — on a typical
    /// in-region AWS S3 path that's a ~25-50 ms saving per
    /// cold open.
    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        if len == 0 {
            // Suffix-range of 0 isn't well-defined in HTTP;
            // fall through to a HEAD so we still return the
            // size for consistency with the default impl.
            let meta = self.head(uri).await?;
            return Ok((Bytes::new(), meta.size));
        }
        let path = self.path(uri)?;
        let tl = io_counters::timeline_start();
        let out = retry::with_reissue(|| async {
            let opts = GetOptions {
                range: Some(GetRange::Suffix(len)),
                ..Default::default()
            };
            let result = self
                .store
                .get_opts(&path, opts)
                .await
                .map_err(|e| translate(uri, e))?;
            let size = result.meta.size as u64;
            let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
            Ok((bytes, size))
        })
        .await;
        if let Ok((b, size)) = &out {
            let start = size.saturating_sub(b.len() as u64);
            self.meter
                .record_get(uri, Some((start, *size)), b.len() as u64);
            io_counters::timeline_record("tail", uri, start, b.len() as u64, tl);
        }
        out
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let path = self.path(uri)?;
        let n = bytes.len() as u64;
        // Re-issue transient failures like the read paths. Only
        // `TransientExhausted` re-issues, so an OCC `PreconditionFailed` still
        // surfaces immediately; a create-only PUT that never landed is safe to retry.
        let out = retry::with_reissue(|| {
            let bytes = bytes.clone();
            async {
                let opts = PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                };
                self.store
                    .put_opts(&path, PutPayload::from_bytes(bytes), opts)
                    .await
                    .map(|r| r.e_tag)
                    .map_err(|e| translate(uri, e))
            }
        })
        .await;
        if out.is_ok() {
            self.meter.record_put(n);
        }
        out
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let path = self.path(uri)?;
        let opts = match expected_etag {
            // None == create-only-if-absent.
            None => PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            },
            // Some(tag) == native S3 conditional update.
            // S3 enforces the etag-match atomically; on
            // conflict the server returns 412 Precondition
            // Failed, which object_store maps to
            // `Error::Precondition` and our translate maps
            // to `StorageError::PreconditionFailed`. No
            // TOCTOU window — the read-then-write that
            // LocalFS needs (and races) is unnecessary here.
            Some(expected) => PutOptions {
                mode: PutMode::Update(UpdateVersion {
                    e_tag: Some(expected.to_string()),
                    version: None,
                }),
                ..Default::default()
            },
        };
        let n = bytes.len() as u64;
        let out = self
            .store
            .put_opts(&path, PutPayload::from_bytes(bytes), opts)
            .await
            .map(|r| r.e_tag)
            .map_err(|e| translate(uri, e));
        if out.is_ok() {
            self.meter.record_put(n);
        }
        out
    }

    async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
        let path = self.path(uri)?;
        let upload = self
            .store
            .put_multipart(&path)
            .await
            .map_err(|e| translate(uri, e))?;
        // CreateMultipartUpload is billable; count only after the session exists.
        self.meter.record_put(0);
        Ok(counting::wrap_multipart(upload, Arc::clone(&self.meter)))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        let path = self.path(uri)?;
        match self.store.delete(&path).await {
            Ok(()) => {
                self.meter.record_delete();
                Ok(())
            }
            // Idempotent delete: NotFound is success for the caller and is
            // still a completed DeleteObject (or equivalent) round-trip.
            Err(ObjError::NotFound { .. }) => {
                self.meter.record_delete();
                Ok(())
            }
            Err(e) => Err(translate(uri, e)),
        }
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        let path = self.path(prefix)?;
        let mut stream = self.store.list(Some(&path));
        // LIST is billable once the stream exists (path already validated);
        // a mid-iteration failure must not undercount the request.
        self.meter.record_list();
        let mut out = Vec::new();
        while let Some(meta) = stream.try_next().await.map_err(|e| translate(prefix, e))? {
            let location = meta.location.to_string();
            out.push((
                logical_list_key(&self.prefix, &location),
                ObjectMeta {
                    size: meta.size,
                    etag: meta.e_tag,
                    last_modified: meta.last_modified.into(),
                },
            ));
        }
        Ok(out)
    }

    fn object_store_handle(&self, uri: &str) -> Option<(Arc<dyn ObjectStore>, ObjPath)> {
        let path = self.path(uri).ok()?;
        Some((
            counting::wrap_object_store(
                Arc::clone(&self.store) as Arc<dyn ObjectStore>,
                Arc::clone(&self.meter),
            ),
            path,
        ))
    }

    fn usage_meter(&self) -> Arc<UsageMeter> {
        Arc::clone(&self.meter)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for error translation, path parsing, and constructors.
    //! `StorageProvider` wire round-trips live in
    //! `tests/supertable/storage/rustfs_s3_wire.rs` (integration test binary —
    //! the lib unit-test graph cannot link `infino_bench_utils` without a second
    //! `infino` crate instance).

    use super::*;

    // ---- translate -----------------------------------------------------

    #[test]
    fn translate_not_found_to_typed_variant() {
        let err = translate(
            "some/key",
            ObjError::NotFound {
                path: "some/key".into(),
                source: "raw".into(),
            },
        );
        match err {
            StorageError::NotFound { uri } => assert_eq!(uri, "some/key"),
            other => panic!("expected NotFound; got {other:?}"),
        }
    }

    #[test]
    fn translate_already_exists_to_precondition_failed() {
        let err = translate(
            "k",
            ObjError::AlreadyExists {
                path: "k".into(),
                source: "raw".into(),
            },
        );
        assert!(matches!(err, StorageError::PreconditionFailed { uri } if uri == "k"));
    }

    #[test]
    fn translate_precondition_to_precondition_failed() {
        let err = translate(
            "k",
            ObjError::Precondition {
                path: "k".into(),
                source: "raw".into(),
            },
        );
        assert!(matches!(err, StorageError::PreconditionFailed { uri } if uri == "k"));
    }

    #[test]
    fn translate_generic_to_transient_exhausted() {
        let err = translate(
            "k",
            ObjError::Generic {
                store: "S3",
                source: "boom".into(),
            },
        );
        match err {
            StorageError::TransientExhausted { uri, .. } => assert_eq!(uri, "k"),
            other => panic!("expected TransientExhausted; got {other:?}"),
        }
    }

    #[test]
    fn translate_other_variant_to_permanent() {
        // Any object_store error variant that isn't one of the
        // explicit arms above maps to Permanent. UnknownConfigurationKey
        // is a stable variant we can construct without an API quirk.
        let err = translate(
            "k",
            ObjError::UnknownConfigurationKey {
                store: "S3",
                key: "foo".into(),
            },
        );
        match err {
            StorageError::Permanent { uri, .. } => assert_eq!(uri, "k"),
            other => panic!("expected Permanent; got {other:?}"),
        }
    }

    // ---- path ----------------------------------------------------------

    #[test]
    fn path_parses_simple_uri() {
        let p = endpoint_provider().path("foo/bar.txt").expect("parse");
        assert_eq!(p.to_string(), "foo/bar.txt");
    }

    #[test]
    fn path_parses_nested_uri() {
        let p = endpoint_provider()
            .path("manifest/manifest-000042.json")
            .expect("parse");
        assert_eq!(p.to_string(), "manifest/manifest-000042.json");
    }

    // ---- constructors --------------------------------------------------

    fn endpoint_provider() -> S3StorageProvider {
        // Pure construction — no I/O. Builds the inner
        // AmazonS3 with explicit credentials targeting a
        // fake endpoint. Useful for testing `bucket()` and
        // `path()` without spinning up the RustFS harness.
        S3StorageProvider::new_with_endpoint(
            "http://127.0.0.1:1",
            "test-bucket",
            "AKIATESTKEY",
            "secret/example",
            "us-east-1",
            None,
        )
        .expect("construct with endpoint")
    }

    #[test]
    fn new_with_endpoint_builds_succeeds_and_exposes_bucket() {
        let p = endpoint_provider();
        assert_eq!(p.bucket(), "test-bucket");
    }

    #[test]
    fn from_object_store_preserves_bucket() {
        // Construct an AmazonS3 directly and wrap it via the
        // escape-hatch constructor. Exercises the wrapping
        // path without going through `new_with_endpoint`'s
        // builder.
        let store = AmazonS3Builder::new()
            .with_endpoint("http://127.0.0.1:1")
            .with_bucket_name("hatch-bucket")
            .with_access_key_id("AKIATESTKEY")
            .with_secret_access_key("secret")
            .with_region("us-east-1")
            .with_allow_http(true)
            .with_virtual_hosted_style_request(false)
            .build()
            .expect("build AmazonS3");
        let p = S3StorageProvider::from_object_store("hatch-bucket", store);
        assert_eq!(p.bucket(), "hatch-bucket");
    }

    #[test]
    fn debug_impl_does_not_panic() {
        // S3StorageProvider derives Debug; print it to ensure
        // the impl block isn't dropped accidentally.
        let p = endpoint_provider();
        let s = format!("{p:?}");
        assert!(s.contains("S3StorageProvider"));
    }

    // ---- pure helpers: prefix / key ------------------------------------

    #[test]
    fn normalize_prefix_trims_surrounding_slashes() {
        assert_eq!(normalize_prefix("/tbl/"), "tbl");
        assert_eq!(normalize_prefix("///a/b///"), "a/b");
        assert_eq!(normalize_prefix("plain"), "plain");
        assert_eq!(normalize_prefix(""), "");
    }

    #[test]
    fn key_without_prefix_strips_leading_slash() {
        let p = endpoint_provider();
        assert_eq!(p.prefix(), "");
        assert_eq!(p.key("/foo/bar"), "foo/bar");
        assert_eq!(p.key("foo/bar"), "foo/bar");
    }

    #[test]
    fn key_with_prefix_prepends_and_strips_leading_slash() {
        let mut p = endpoint_provider();
        p.prefix = "tbl".into();
        assert_eq!(p.prefix(), "tbl");
        assert_eq!(p.key("data/seg-1"), "tbl/data/seg-1");
        assert_eq!(p.key("/data/seg-1"), "tbl/data/seg-1");
    }

    #[test]
    fn new_with_endpoint_and_prefix_normalizes_and_applies_prefix() {
        let p = S3StorageProvider::new_with_endpoint_and_prefix(
            "http://127.0.0.1:1",
            "b",
            "AKIATESTKEY",
            "secret",
            "us-east-1",
            "/scoped/tbl/",
            None,
        )
        .expect("construct with endpoint + prefix");
        assert_eq!(p.bucket(), "b");
        assert_eq!(p.prefix(), "scoped/tbl");
        assert_eq!(p.key("data/seg-1"), "scoped/tbl/data/seg-1");
    }

    #[test]
    fn object_store_handle_returns_path_under_prefix() {
        let mut p = endpoint_provider();
        p.prefix = "tbl".into();
        let (_, path) = p
            .object_store_handle("data/seg-1")
            .expect("handle for valid uri");
        assert_eq!(path.to_string(), "tbl/data/seg-1");
    }
}
