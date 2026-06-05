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
//! Three shapes, all behind the same [`Self::new`] +
//! `*_with_endpoint` constructors:
//!
//!   - **AWS production**: build the underlying
//!     `AmazonS3Builder` from environment (AWS_ACCESS_KEY_ID
//!     etc.) and pass it via [`Self::from_object_store`].
//!   - **s3s-fs test harness**: [`Self::new_with_endpoint`]
//!     takes the harness's `http://127.0.0.1:<port>` endpoint
//!     plus a bucket name + test credential pair. The
//!     `supertable/storage/smoke_s3.rs` integration test uses
//!     this to exercise the wire protocol without an AWS
//!     account.
//!   - **Self-hosted S3-compatible** (Ceph, R2, etc.): same
//!     `new_with_endpoint` shape with the relevant endpoint +
//!     credentials.

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::TryStreamExt;
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::path::Path as ObjPath;
use object_store::{
    Error as ObjError, GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions,
    PutPayload, UpdateVersion,
};

use super::{ObjectMeta, StorageError, StorageProvider};

/// S3-backed `StorageProvider`. Cheap to clone; the inner
/// `AmazonS3` shares its HTTP client across clones.
#[derive(Debug)]
pub struct S3StorageProvider {
    bucket: String,
    prefix: String,
    store: Arc<AmazonS3>,
}

impl S3StorageProvider {
    /// Construct an S3 provider from the standard AWS
    /// credential chain (env vars / instance profile / etc.)
    /// + an explicit bucket. The supertable's URIs are
    /// keyed off `<bucket>/<uri>`.
    pub fn new(bucket: impl Into<String>) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch)
            .with_client_options(tuned_client_options())
            .with_retry(tuned_retry_config())
            .build()
            .map_err(|e| StorageError::Permanent {
                uri: format!("s3://{bucket}"),
                source: Box::new(e),
            })?;
        Ok(Self {
            bucket,
            prefix: String::new(),
            store: Arc::new(store),
        })
    }

    /// Construct an S3 provider scoped to a logical table
    /// prefix inside `bucket`. The prefix is prepended to every
    /// storage URI, so callers can use the normal supertable
    /// paths (`_supertable/current`, `data/seg-...`) while
    /// isolating each table under `s3://bucket/prefix/`.
    pub fn new_with_prefix(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let mut provider = Self::new(bucket)?;
        provider.prefix = normalize_prefix(prefix);
        Ok(provider)
    }

    /// Construct an S3 provider pointed at a custom endpoint
    /// + explicit credentials. Used by
    /// `tests/supertable_smoke_s3.rs` for the s3s-fs
    /// integration test (`endpoint = "http://127.0.0.1:<port>"`)
    /// and by callers using a self-hosted S3-compatible
    /// service (MinIO etc.).
    ///
    /// `allow_http` is enabled so plain-HTTP endpoints
    /// (typical for in-process test harnesses) don't get
    /// rejected by the AWS SDK's HTTPS check.
    pub fn new_with_endpoint(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let endpoint = endpoint.into();
        let store = AmazonS3Builder::new()
            .with_endpoint(endpoint.clone())
            .with_bucket_name(&bucket)
            .with_access_key_id(access_key.into())
            .with_secret_access_key(secret_key.into())
            .with_region(region.into())
            .with_allow_http(true)
            // Force path-style addressing (bucket as path
            // prefix, not subdomain). Required for
            // localhost-style endpoints (s3s-fs, MinIO,
            // any non-AWS S3-compatible service that
            // doesn't terminate `<bucket>.<endpoint>` DNS).
            .with_virtual_hosted_style_request(false)
            .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch)
            // NB: do NOT apply `tuned_client_options()` here. The
            // deep idle-connection pool / long keep-alive is tuned
            // for real-S3 fan-out latency and destabilizes local
            // S3-compatible endpoints (s3s-fs / MinIO): reqwest
            // reuses connections the emulator has already closed,
            // surfacing as "error sending request". Also,
            // `with_client_options` would clobber the
            // `with_allow_http(true)` above. The endpoint path keeps
            // object_store's defaults.
            .build()
            .map_err(|e| StorageError::Permanent {
                uri: format!("s3://{bucket} @ {endpoint}"),
                source: Box::new(e),
            })?;
        Ok(Self {
            bucket,
            prefix: String::new(),
            store: Arc::new(store),
        })
    }

    /// Custom-endpoint variant of [`Self::new_with_prefix`].
    /// Used by S3-compatible deployments that also want a
    /// logical table prefix.
    pub fn new_with_endpoint_and_prefix(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let mut provider =
            Self::new_with_endpoint(endpoint, bucket, access_key, secret_key, region)?;
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

/// Tuned HTTP client options for the object-store-native fan-out.
///
/// The supertable vector/FTS query path fans out one cold-open +
/// cold-search batch per segment concurrently. With the default
/// idle-connection pool, a wide fan-out (hundreds of segments ×
/// several range GETs each) churns TCP/TLS connections — each new
/// connection pays a TLS handshake RTT on top of the request RTT,
/// inflating the p99 tail under load. Keeping a large warm idle
/// pool lets the fan-out reuse connections so the per-GET cost is
/// one RTT, not handshake + RTT.
fn tuned_client_options() -> object_store::ClientOptions {
    object_store::ClientOptions::new()
        // Keep many connections warm per host so concurrent
        // fan-out GETs reuse established TLS sessions instead of
        // handshaking. AWS S3 in-region serves many parallel
        // range GETs per host; a deep idle pool is the difference
        // between "RTT" and "handshake + RTT" on the cold tail.
        .with_pool_max_idle_per_host(1024)
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
        .with_pool_idle_timeout(std::time::Duration::from_secs(10))
        // Bound the connect phase so a single slow SYN/TLS doesn't
        // dominate the fan-out's p99; the retry layer covers drops.
        .with_connect_timeout(std::time::Duration::from_secs(5))
}

/// Tuned retry budget for real-S3 fan-out.
///
/// The cold vector/FTS query path fires a burst of hundreds of
/// concurrent range GETs per query. A single wave can momentarily
/// trip a transient transport error (a connection reset, or an
/// `error sending request` on a socket S3 closed under us) that
/// `object_store`'s retry layer would otherwise exhaust — surfacing
/// as `TransientExhausted` straight into the query. A deeper budget
/// (and a longer overall window) lets the client ride out a burst
/// instead of failing the query. Paired with the sub-S3-idle-close
/// `pool_idle_timeout` above, which removes the dominant cause of
/// those errors in the first place.
fn tuned_retry_config() -> object_store::RetryConfig {
    object_store::RetryConfig {
        max_retries: 20,
        retry_timeout: std::time::Duration::from_secs(300),
        ..Default::default()
    }
}

/// Application-level re-issue budget for a transient transport
/// failure on a range GET (e.g. reqwest "error sending request"
/// that `object_store` does not retry itself). Each attempt dials
/// a fresh connection, so a healthy host recovers within a couple
/// of tries; the cap bounds a genuinely-unreachable endpoint.
const MAX_TRANSIENT_RETRIES: u32 = 8;

/// `true` for storage errors worth re-issuing an idempotent GET
/// against — transport-level flakiness that cleared object_store's
/// own (ineffective for this class) retry without succeeding.
/// `NotFound` / `PreconditionFailed` / `Permanent` are stable and
/// never retried here.
fn is_retryable_transient(err: &StorageError) -> bool {
    matches!(err, StorageError::TransientExhausted { .. })
}

/// Exponential backoff for transient GET retries: 50ms, 100, 200,
/// 400, ... capped at 2s. Brief by design — the goal is to let the
/// dead pooled connection drain and a fresh dial succeed, not to
/// wait out a long outage.
fn transient_backoff(attempt: u32) -> std::time::Duration {
    let ms = 50u64.saturating_mul(1 << attempt.min(5));
    std::time::Duration::from_millis(ms.min(2000))
}

/// Definitive error for a range the backing object can't satisfy
/// (it returned fewer bytes than requested and made no further
/// progress). Surfaced as `Permanent` — callers do not retry; a
/// shorter-than-requested object is a stable condition, not a
/// transient one.
fn short_read(uri: &str, start: u64, requested: u64, got: u64) -> StorageError {
    let boxed: Box<dyn std::error::Error + Send + Sync> = format!(
        "get_range short read: object returned {got} of {requested} bytes from offset {start}"
    )
    .into();
    StorageError::Permanent {
        uri: uri.into(),
        source: boxed,
    }
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
        })
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = self.path(uri)?;
        let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
        // `GetResult.meta` is the version whose bytes we're
        // about to read — etag and bytes are atomically paired
        // by S3, so no follow-up HEAD is needed.
        let meta = ObjectMeta {
            size: result.meta.size as u64,
            etag: result.meta.e_tag.clone(),
        };
        let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
        Ok((bytes, meta))
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = self.path(uri)?;
        let want = range.end.saturating_sub(range.start);
        if want == 0 {
            return Ok(Bytes::new());
        }

        // Completion loop. `object_store::get_range` is expected to
        // return exactly the requested span, but an S3 GET can come
        // back *short* without erroring — a body truncated by a
        // transient transport hiccup. A short buffer that slips
        // through here corrupts callers in two ways: the foreground
        // lazy reader slices past the end (panic), and the background
        // cache fill `pwrite`s it at the chunk offset, leaving a zero
        // gap in the mmap'd file. Re-issue the GET for the
        // still-missing tail; surface a definitive error only when
        // the object genuinely can't satisfy the range.
        let mut cursor = range.start;
        let mut filled: u64 = 0;
        let mut parts: Vec<Bytes> = Vec::new();
        let mut transient_retries = 0u32;
        loop {
            // Application-level retry for transient connection/transport
            // failures. `object_store`'s own retry layer does NOT cover
            // reqwest "error sending request" (a send-side failure on a
            // socket the server closed under us): it gives up in
            // milliseconds, well inside its configured window, and
            // surfaces the error straight into the query. Under the cold
            // vector/FTS fan-out (a burst of hundreds of concurrent GETs)
            // those send failures are common. A range GET is idempotent,
            // so re-issuing the still-missing tail with backoff is safe;
            // a fresh attempt also forces reqwest to discard the dead
            // pooled connection and dial a new one.
            let chunk = match self.store.get_range(&path, cursor..range.end).await {
                Ok(chunk) => chunk,
                Err(e) => {
                    let err = translate(uri, e);
                    if is_retryable_transient(&err) && transient_retries < MAX_TRANSIENT_RETRIES {
                        tokio::time::sleep(transient_backoff(transient_retries)).await;
                        transient_retries += 1;
                        continue;
                    }
                    return Err(err);
                }
            };
            if chunk.is_empty() {
                // A zero-byte body for an *in-bounds* range is a
                // transport glitch (a reset/truncated response delivered
                // as a 200/206 with no payload), NOT a real end-of-object
                // — object_store surfaces a genuinely past-the-end range
                // as a typed `NotFound`/`Generic` error, never as an empty
                // `Ok`. The original bug here treated the first empty reply
                // as a permanent short read, which aborted cold queries
                // mid-fan-out at scale (e.g. 0 of 120576 bytes at an offset
                // well inside a 59 MiB segment). Re-issue with backoff (which
                // also forces a fresh connection) and only surface the
                // definitive short read once the transient budget is spent.
                if transient_retries < MAX_TRANSIENT_RETRIES {
                    tokio::time::sleep(transient_backoff(transient_retries)).await;
                    transient_retries += 1;
                    continue;
                }
                return Err(short_read(uri, range.start, want, filled));
            }
            let take = (chunk.len() as u64).min(want - filled);
            filled += take;
            cursor += take;
            if take as usize == chunk.len() {
                parts.push(chunk);
            } else {
                parts.push(chunk.slice(0..take as usize));
            }
            if filled >= want {
                break;
            }
            // A non-empty chunk shorter than requested is normal for a
            // large range (S3 may split it): `filled` strictly advances
            // every iteration, so the loop is bounded and terminates.
            // No-progress (empty) iterations are handled above via the
            // transient-retry budget, so there is no separate stall cap.
        }

        // Fast path: a single full-length response is zero-copy.
        if parts.len() == 1 {
            return Ok(parts.pop().expect("len checked == 1"));
        }
        let mut out = BytesMut::with_capacity(want as usize);
        for p in &parts {
            out.extend_from_slice(p);
        }
        Ok(out.freeze())
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
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let path = self.path(uri)?;
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

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
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

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let path = ObjPath::from(prefix);
        let mut stream = self.store.list(Some(&path));
        let mut out: Vec<String> = Vec::new();
        while let Some(meta) = stream.try_next().await.map_err(|e| translate(prefix, e))? {
            out.push(meta.location.to_string());
        }
        Ok(out)
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
}
