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
    /// Construct an S3 provider from the standard AWS
    /// credential chain (env vars / instance profile / etc.)
    /// + an explicit bucket. The supertable's URIs are
    /// keyed off `<bucket>/<uri>`.
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
