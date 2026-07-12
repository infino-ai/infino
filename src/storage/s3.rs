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
//! [`Self::new_with_endpoint`] is a convenience for s3s-fs / MinIO / Ceph.

use std::{ops::Range, str::FromStr, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{
    ClientOptions, Error as ObjError, GetOptions, GetRange, MultipartUpload, ObjectStore,
    ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion,
    aws::{AmazonS3, AmazonS3Builder, AmazonS3ConfigKey, S3ConditionalPut},
    path::Path as ObjPath,
};

use super::{ObjectMeta, StorageError, StorageOptions, StorageProvider, options::apply, retry};

/// Config key written by [`S3StorageProvider::new_with_endpoint`] to point
/// at a custom endpoint. Detection accepts any object_store alias (see
/// [`has_custom_endpoint`]); this is just the canonical name to set.
const ENDPOINT_KEY: &str = "aws_endpoint";

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
    /// destabilizes local s3s-fs / MinIO endpoints).
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
        })
    }

    /// Custom S3-compatible endpoint with static credentials (s3s-fs /
    /// MinIO / Ceph). `allow_http` is enabled for plain-HTTP endpoints.
    pub fn new_with_endpoint(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Result<Self, StorageError> {
        Self::new_with_endpoint_and_prefix(endpoint, bucket, access_key, secret_key, region, "")
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
    ) -> Result<Self, StorageError> {
        let opts = StorageOptions::from([
            (ENDPOINT_KEY.to_string(), endpoint.into()),
            ("aws_access_key_id".to_string(), access_key.into()),
            ("aws_secret_access_key".to_string(), secret_key.into()),
            ("aws_region".to_string(), region.into()),
            ("aws_allow_http".to_string(), "true".to_string()),
        ]);
        Self::new_with_prefix(bucket, prefix, &opts)
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
        }
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
        Ok(ObjectMeta {
            size: meta.size as u64,
            etag: meta.e_tag,
            last_modified: meta.last_modified.into(),
        })
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = self.path(uri)?;
        // etag and bytes are atomically paired in the same response, so
        // no follow-up HEAD is needed.
        retry::with_reissue(|| async {
            let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
            let meta = ObjectMeta {
                size: result.meta.size as u64,
                etag: result.meta.e_tag.clone(),
                last_modified: result.meta.last_modified.into(),
            };
            let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
            Ok((bytes, meta))
        })
        .await
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(uri = uri, len = range.end - range.start))
    )]
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = self.path(uri)?;
        retry::complete_range(uri, range, |r| async {
            self.store
                .get_range(&path, r)
                .await
                .map_err(|e| translate(uri, e))
        })
        .await
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
        retry::with_reissue(|| async {
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
        .await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let path = self.path(uri)?;
        // Re-issue transient failures like the read paths. Only
        // `TransientExhausted` re-issues, so an OCC `PreconditionFailed` still
        // surfaces immediately; a create-only PUT that never landed is safe to retry.
        retry::with_reissue(|| {
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
        .await
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
        self.store
            .put_opts(&path, PutPayload::from_bytes(bytes), opts)
            .await
            .map(|r| r.e_tag)
            .map_err(|e| translate(uri, e))
    }

    async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
        let path = self.path(uri)?;
        self.store
            .put_multipart(&path)
            .await
            .map_err(|e| translate(uri, e))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        let path = self.path(uri)?;
        match self.store.delete(&path).await {
            Ok(()) => Ok(()),
            Err(ObjError::NotFound { .. }) => Ok(()),
            Err(e) => Err(translate(uri, e)),
        }
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        let path = ObjPath::from(prefix);
        let mut stream = self.store.list(Some(&path));
        let mut out = Vec::new();
        while let Some(meta) = stream.try_next().await.map_err(|e| translate(prefix, e))? {
            out.push((
                meta.location.to_string(),
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
        Some((Arc::clone(&self.store) as Arc<dyn ObjectStore>, path))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the parts of `s3.rs` that don't need a
    //! real HTTP backend: error translation, path parsing,
    //! the with-endpoint constructor, and the `from_object_store`
    //! escape hatch. The trait impls (`head`, `get`, `put_*`,
    //! `delete`, `get_range`) are exercised end-to-end by the
    //! `supertable_smoke_via_s3_wire_protocol` integration
    //! test against an in-process `s3s-fs` server.
    //!
    //! In addition, this module stands up the same in-process
    //! `s3s-fs` server the smoke test uses (the dev-dependencies
    //! `s3s` / `s3s-fs` / `hyper-util` are on the lib unit-test
    //! compile graph), so the trait impls — `put_atomic`, `get`,
    //! `get_range`, `head`, `delete`, `list_with_prefix`,
    //! `put_if_match`, and `tail` — are exercised here over the
    //! real S3 HTTP wire protocol without any cloud credentials
    //! or network access.
    use std::{fs::create_dir_all, net::SocketAddr};

    use s3s::{auth::SimpleAuth, service::S3ServiceBuilder};
    use s3s_fs::FileSystem;
    use tempfile::TempDir;
    use tokio::net::TcpListener;

    use super::*;

    // ---- in-process s3s-fs harness -------------------------------------

    /// Bucket the in-process server pre-creates for round-trip tests.
    const HARNESS_BUCKET: &str = "infino-s3-unit";
    /// Region passed to the provider; arbitrary for s3s-fs.
    const HARNESS_REGION: &str = "us-east-1";
    /// Fixed dummy credential pair s3s validates SigV4 against.
    const HARNESS_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
    const HARNESS_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

    /// Spawn s3s-fs on a random loopback port. Returns the bound
    /// address plus the tempdir guard (drop unlinks the bucket data,
    /// so the caller must keep it alive for the test's duration).
    async fn spawn_s3s_fs() -> (SocketAddr, TempDir) {
        let fs_root = TempDir::new().expect("s3s-fs root tempdir");
        // s3s-fs treats top-level dirs as buckets; pre-create the
        // bucket dir so a put on a key inside it doesn't 404 the
        // bucket.
        create_dir_all(fs_root.path().join(HARNESS_BUCKET)).expect("create bucket dir");

        let fs_backend = FileSystem::new(fs_root.path()).expect("s3s-fs FileSystem");
        let service = {
            let mut b = S3ServiceBuilder::new(fs_backend);
            // Without an auth provider s3s answers 501 to any signed
            // request; object_store always signs.
            b.set_auth(SimpleAuth::from_single(
                HARNESS_ACCESS_KEY,
                HARNESS_SECRET_KEY,
            ));
            b.build()
        };

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

    /// Build a provider pointed at a freshly spawned in-process server.
    /// Returns the provider plus the tempdir guard.
    async fn harness_provider() -> (S3StorageProvider, TempDir) {
        let (addr, guard) = spawn_s3s_fs().await;
        let endpoint = format!("http://{addr}");
        let provider = S3StorageProvider::new_with_endpoint(
            endpoint,
            HARNESS_BUCKET,
            HARNESS_ACCESS_KEY,
            HARNESS_SECRET_KEY,
            HARNESS_REGION,
        )
        .expect("construct provider against in-process s3s-fs");
        (provider, guard)
    }

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
            .path("manifest-lists/list-000042.json")
            .expect("parse");
        assert_eq!(p.to_string(), "manifest-lists/list-000042.json");
    }

    // ---- constructors --------------------------------------------------

    fn endpoint_provider() -> S3StorageProvider {
        // Pure construction — no I/O. Builds the inner
        // AmazonS3 with explicit credentials targeting a
        // fake endpoint. Useful for testing `bucket()` and
        // `path()` without spinning up the s3s-fs harness.
        S3StorageProvider::new_with_endpoint(
            "http://127.0.0.1:1",
            "test-bucket",
            "AKIATESTKEY",
            "secret/example",
            "us-east-1",
        )
        .expect("construct with endpoint")
    }

    #[test]
    fn new_with_endpoint_builds_succeeds_and_exposes_bucket() {
        let p = endpoint_provider();
        assert_eq!(p.bucket(), "test-bucket");
    }

    #[test]
    fn rejects_unknown_storage_option_key() {
        let opts = StorageOptions::from([("not_a_real_key".to_string(), "x".to_string())]);
        let err = S3StorageProvider::new_with_prefix("b", "", &opts).expect_err("bad key");
        assert!(matches!(err, StorageError::Permanent { .. }));
    }

    #[test]
    fn rejects_cross_backend_azure_key() {
        let opts =
            StorageOptions::from([("azure_storage_account_name".to_string(), "acct".to_string())]);
        assert!(S3StorageProvider::new_with_prefix("b", "", &opts).is_err());
    }

    #[test]
    fn detects_custom_endpoint_under_any_alias() {
        for key in [
            "aws_endpoint",
            "endpoint",
            "aws_endpoint_url",
            "aws_endpoint_url_s3",
        ] {
            let opts = StorageOptions::from([(key.to_string(), "http://localhost".to_string())]);
            assert!(
                has_custom_endpoint(&opts),
                "{key} should select the endpoint profile"
            );
        }
    }

    #[test]
    fn no_endpoint_for_credentials_only() {
        let opts = StorageOptions::from([("aws_region".to_string(), "us-east-1".to_string())]);
        assert!(!has_custom_endpoint(&opts));
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

    // ---- in-process s3s-fs round-trips ---------------------------------
    //
    // These exercise the StorageProvider trait impls over the real S3
    // HTTP wire protocol against the in-process s3s-fs server — no cloud
    // credentials, no external network.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn put_atomic_then_get_round_trips() {
        let (p, _guard) = harness_provider().await;
        let body = Bytes::from_static(b"hello-unit-s3");
        p.put_atomic("k/hello.txt", body.clone())
            .await
            .expect("put_atomic");
        let (got, meta) = p.get("k/hello.txt").await.expect("get");
        assert_eq!(got, body);
        assert_eq!(meta.size, body.len() as u64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cas_conformance_holds() {
        // s3s-fs does not enforce a stale conditional update (its 412 path is
        // covered by the real-S3 integration smoke), so stale rejection is
        // not asserted here — the chained-token step is what matters.
        let (p, _guard) = harness_provider().await;
        crate::test_helpers::cas_conformance::cas_conformance(&p, "cas/conf", false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn put_atomic_twice_is_precondition_failed() {
        let (p, _guard) = harness_provider().await;
        let body = Bytes::from_static(b"first");
        p.put_atomic("k/dup", body.clone())
            .await
            .expect("first put");
        // PutMode::Create on an existing key -> 412/conflict ->
        // PreconditionFailed.
        let err = p
            .put_atomic("k/dup", Bytes::from_static(b"second"))
            .await
            .expect_err("second create must fail");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed; got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_missing_is_not_found() {
        let (p, _guard) = harness_provider().await;
        let err = p.get("k/absent").await.expect_err("get missing must fail");
        assert!(
            matches!(err, StorageError::NotFound { .. }),
            "expected NotFound; got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn head_missing_is_not_found() {
        let (p, _guard) = harness_provider().await;
        let err = p.head("k/absent").await.expect_err("head missing fails");
        assert!(
            matches!(err, StorageError::NotFound { .. }),
            "expected NotFound; got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn head_reports_size() {
        let (p, _guard) = harness_provider().await;
        let body = Bytes::from_static(b"0123456789");
        p.put_atomic("k/sized", body.clone())
            .await
            .expect("put_atomic");
        let meta = p.head("k/sized").await.expect("head");
        assert_eq!(meta.size, body.len() as u64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_range_returns_subslice() {
        let (p, _guard) = harness_provider().await;
        let body: Vec<u8> = (0..=255u8).collect();
        p.put_atomic("k/range.bin", Bytes::from(body.clone()))
            .await
            .expect("put_atomic");
        let got = p.get_range("k/range.bin", 10..20).await.expect("get_range");
        assert_eq!(&got[..], &body[10..20]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tail_returns_trailing_bytes_and_size() {
        let (p, _guard) = harness_provider().await;
        let body: Vec<u8> = (0..200u8).collect();
        p.put_atomic("k/tail.bin", Bytes::from(body.clone()))
            .await
            .expect("put_atomic");
        // S3 path uses a native suffix-range fetch.
        let (tail, size) = p.tail("k/tail.bin", 32).await.expect("tail");
        assert_eq!(size, body.len() as u64);
        assert_eq!(&tail[..], &body[body.len() - 32..]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tail_zero_len_falls_back_to_head_for_size() {
        let (p, _guard) = harness_provider().await;
        let body = Bytes::from_static(b"abcdef");
        p.put_atomic("k/tail0.bin", body.clone())
            .await
            .expect("put_atomic");
        let (tail, size) = p.tail("k/tail0.bin", 0).await.expect("zero-len tail");
        assert!(tail.is_empty());
        assert_eq!(size, body.len() as u64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_removes_object() {
        let (p, _guard) = harness_provider().await;
        p.put_atomic("k/del", Bytes::from_static(b"x"))
            .await
            .expect("put_atomic");
        p.delete("k/del").await.expect("delete existing");
        let err = p.get("k/del").await.expect_err("deleted object gone");
        assert!(matches!(err, StorageError::NotFound { .. }));
        // NB: the delete-absent idempotency (Err(NotFound) => Ok arm) is
        // not asserted here — s3s-fs returns a malformed delete response
        // for a no-op delete, which is an emulator quirk, not a code path
        // we own. Real S3 returns 204 for an absent key.
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_with_prefix_returns_matching_keys() {
        let (p, _guard) = harness_provider().await;
        p.put_atomic("list/a.txt", Bytes::from_static(b"a"))
            .await
            .expect("put a");
        p.put_atomic("list/b.txt", Bytes::from_static(b"b"))
            .await
            .expect("put b");
        p.put_atomic("other/c.txt", Bytes::from_static(b"c"))
            .await
            .expect("put c");
        let mut keys = p.list_with_prefix("list/").await.expect("list");
        keys.sort();
        assert_eq!(
            keys,
            vec!["list/a.txt".to_string(), "list/b.txt".to_string()]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn put_if_match_none_is_create_only() {
        let (p, _guard) = harness_provider().await;
        // First create-if-absent succeeds.
        p.put_if_match("k/cas", Bytes::from_static(b"v1"), None)
            .await
            .expect("create-if-absent");
        // Second create-if-absent on the same key conflicts.
        let err = p
            .put_if_match("k/cas", Bytes::from_static(b"v2"), None)
            .await
            .expect_err("second create-if-absent must fail");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed; got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn put_if_match_etag_update_succeeds_with_matching_etag() {
        let (p, _guard) = harness_provider().await;
        let etag = p
            .put_atomic("k/upd", Bytes::from_static(b"v1"))
            .await
            .expect("initial put")
            .expect("s3 returns an etag on create");
        // Conditional update carrying the current etag succeeds and the
        // new body is visible. (s3s-fs does not enforce the etag-match on
        // a stale conditional update, so the 412/PreconditionFailed arm is
        // covered against real S3 by the integration smoke; here we assert
        // the matching-etag success path, which the emulator does honor.)
        p.put_if_match("k/upd", Bytes::from_static(b"v2"), Some(&etag))
            .await
            .expect("update with matching etag");
        let (got, _) = p.get("k/upd").await.expect("get latest");
        assert_eq!(&got[..], b"v2");
    }
}
