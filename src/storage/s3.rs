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
use bytes::Bytes;
use object_store::aws::{AmazonS3, AmazonS3Builder};
use object_store::path::Path as ObjPath;
use object_store::{
    Error as ObjError, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion,
};

use super::{ObjectMeta, StorageError, StorageProvider};

/// S3-backed `StorageProvider`. Cheap to clone; the inner
/// `AmazonS3` shares its HTTP client across clones.
#[derive(Debug)]
pub struct S3StorageProvider {
    bucket: String,
    store: Arc<AmazonS3>,
}

impl S3StorageProvider {
    /// Construct an S3 provider from the standard AWS credential chain
    /// (env vars / instance profile / etc.) + an explicit bucket. The
    /// supertable's URIs are keyed off `<bucket>/<uri>`.
    pub fn new(bucket: impl Into<String>) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch)
            .build()
            .map_err(|e| StorageError::Permanent {
                uri: format!("s3://{bucket}"),
                source: Box::new(e),
            })?;
        Ok(Self {
            bucket,
            store: Arc::new(store),
        })
    }

    /// Construct an S3 provider pointed at a custom endpoint + explicit
    /// credentials. Used by `tests/supertable_smoke_s3.rs` for the s3s-fs
    /// integration test (`endpoint = "http://127.0.0.1:<port>"`) and by
    /// callers using a self-hosted S3-compatible service (MinIO etc.).
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
            .build()
            .map_err(|e| StorageError::Permanent {
                uri: format!("s3://{bucket} @ {endpoint}"),
                source: Box::new(e),
            })?;
        Ok(Self {
            bucket,
            store: Arc::new(store),
        })
    }

    /// Wrap an already-constructed `AmazonS3` — for advanced
    /// callers that want full control over the
    /// `AmazonS3Builder` (custom retry config, virtual-hosted
    /// vs path-style addressing, etc.).
    pub fn from_object_store(bucket: impl Into<String>, store: AmazonS3) -> Self {
        Self {
            bucket: bucket.into(),
            store: Arc::new(store),
        }
    }

    /// S3 bucket this provider is scoped to.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    fn path(uri: &str) -> Result<ObjPath, StorageError> {
        ObjPath::parse(uri).map_err(|e| StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(e),
        })
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
        let path = Self::path(uri)?;
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

    async fn get(&self, uri: &str) -> Result<Bytes, StorageError> {
        let path = Self::path(uri)?;
        let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
        result.bytes().await.map_err(|e| translate(uri, e))
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = Self::path(uri)?;
        self.store
            .get_range(&path, range)
            .await
            .map_err(|e| translate(uri, e))
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        let path = Self::path(uri)?;
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        self.store
            .put_opts(&path, PutPayload::from_bytes(bytes), opts)
            .await
            .map(|_| ())
            .map_err(|e| translate(uri, e))
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<(), StorageError> {
        let path = Self::path(uri)?;
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
            .map(|_| ())
            .map_err(|e| translate(uri, e))
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        let path = Self::path(uri)?;
        self.store
            .put_multipart(&path)
            .await
            .map_err(|e| translate(uri, e))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        let path = Self::path(uri)?;
        match self.store.delete(&path).await {
            Ok(()) => Ok(()),
            Err(ObjError::NotFound { .. }) => Ok(()),
            Err(e) => Err(translate(uri, e)),
        }
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
        let p = S3StorageProvider::path("foo/bar.txt").expect("parse");
        assert_eq!(p.to_string(), "foo/bar.txt");
    }

    #[test]
    fn path_parses_nested_uri() {
        let p = S3StorageProvider::path("manifest-lists/list-000042.json").expect("parse");
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
