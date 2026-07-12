// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Google Cloud Storage-backed [`StorageProvider`].
//!
//! Wraps `object_store::gcp::GoogleCloudStorage` so the supertable
//! exercises the same code paths on GCS as on S3, Azure, and LocalFS.
//! Conditional writes are native (`x-goog-if-generation-match`) with no
//! builder flag. The one twist vs. S3/Azure: GCS keys conditional updates
//! on the object *generation*, not the HTTP ETag, so this provider carries
//! the generation in [`ObjectMeta::etag`] (an opaque version token) and
//! returns it through `UpdateVersion::version`.

use std::{ops::Range, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{
    ClientOptions, Error as ObjError, GetOptions, GetRange, MultipartUpload, ObjectMeta as OsMeta,
    ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, StaticCredentialProvider,
    UpdateVersion,
    gcp::{GcpCredential, GoogleCloudStorage, GoogleCloudStorageBuilder, GoogleConfigKey},
    path::Path as ObjPath,
};

use super::{ObjectMeta, StorageError, StorageOptions, StorageProvider, options::apply, retry};

/// Warm idle connections per host, so a wide range-GET fan-out reuses TLS
/// sessions instead of re-handshaking on the cold tail.
const GCS_POOL_MAX_IDLE_PER_HOST: usize = 1024;

/// Idle-connection keep-alive, below GCS's server-side close window.
const GCS_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Connect-phase timeout, so one slow SYN/TLS can't dominate the p99.
const GCS_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Whole-request timeout (incl. body), sized for a multi-MB superfile PUT.
const GCS_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Reserved storage option carrying a pre-obtained OAuth2 bearer token for GCS
/// (for example a short-lived, prefix-scoped access token the caller obtained
/// out of band). Unlike the S3 session token or an Azure SAS, a GCS bearer token has no
/// `object_store` config-key, so it is applied programmatically through the
/// client's credential provider. When present it is consumed here rather than
/// folded on as a `google_*` config key (which would be rejected as unknown).
const GCS_BEARER_TOKEN_OPTION: &str = "google_bearer_token";

/// GCS-backed `StorageProvider`. Cheap to clone; the inner
/// `GoogleCloudStorage` shares its HTTP client across clones.
#[derive(Debug)]
pub struct GcsStorageProvider {
    bucket: String,
    prefix: String,
    store: Arc<GoogleCloudStorage>,
}

impl GcsStorageProvider {
    /// GCS provider for `bucket` with no explicit options — credentials
    /// resolve through object_store's ambient chain (GCE metadata /
    /// workload-identity ADC). Infino never reads GCS credentials from the
    /// process environment; pass them through [`Self::new_with_prefix`].
    pub fn new(bucket: impl Into<String>) -> Result<Self, StorageError> {
        Self::new_with_prefix(bucket, "", &StorageOptions::new())
    }

    /// GCS provider scoped to `prefix` inside `bucket`, configured from `opts`:
    /// object_store's `google_*` config strings (service-account key/path), plus
    /// the reserved [`GCS_BEARER_TOKEN_OPTION`] carrying a pre-obtained OAuth2
    /// bearer token. The prefix isolates each table under `gs://bucket/prefix/`.
    pub fn new_with_prefix(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        opts: &StorageOptions,
    ) -> Result<Self, StorageError> {
        let bucket = bucket.into();
        let uri = format!("gs://{bucket}");
        let mut builder = GoogleCloudStorageBuilder::new()
            .with_bucket_name(&bucket)
            .with_client_options(tuned_client_options())
            .with_retry(retry::config());
        // A bearer token has no object_store config-key, so pull it out and set
        // it on the client's credential provider; the rest fold on as config keys.
        let mut config_opts = opts.clone();
        if let Some(bearer) = config_opts.remove(GCS_BEARER_TOKEN_OPTION) {
            builder =
                builder.with_credentials(Arc::new(StaticCredentialProvider::new(GcpCredential {
                    bearer,
                })));
        }
        let builder = apply::<GoogleConfigKey, _>(builder, &config_opts, &uri, |b, key, value| {
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

    /// Wrap an already-constructed `GoogleCloudStorage` — for callers that
    /// want full control over the `GoogleCloudStorageBuilder`.
    pub fn from_object_store(bucket: impl Into<String>, store: GoogleCloudStorage) -> Self {
        Self {
            bucket: bucket.into(),
            prefix: String::new(),
            store: Arc::new(store),
        }
    }

    /// GCS bucket this provider is scoped to.
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
        ObjPath::parse(self.key(uri)).map_err(|e| StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(e),
        })
    }
}

fn normalize_prefix(prefix: impl Into<String>) -> String {
    prefix.into().trim_matches('/').to_string()
}

/// The opaque CAS token our `ObjectMeta.etag` carries for GCS: the object
/// *generation* (what `x-goog-if-generation-match` compares). Never the HTTP
/// ETag — the etag and the generation are different kinds, and coercing one
/// into the other's slot silently breaks conditional writes. Absent
/// generation => `None` (create-only semantics), not a fallback.
fn version_token(meta: &OsMeta) -> Option<String> {
    meta.version.clone()
}

/// HTTP client options: deep warm idle pool + bounded connect/request.
fn tuned_client_options() -> ClientOptions {
    ClientOptions::new()
        .with_pool_max_idle_per_host(GCS_POOL_MAX_IDLE_PER_HOST)
        .with_pool_idle_timeout(GCS_POOL_IDLE_TIMEOUT)
        .with_connect_timeout(GCS_CONNECT_TIMEOUT)
        .with_timeout(GCS_REQUEST_TIMEOUT)
}

/// Translate an `object_store::Error` to our `StorageError`. Kept
/// per-backend (not shared) so its mapping can diverge if GCS's error
/// surface widens.
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
impl StorageProvider for GcsStorageProvider {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        let path = self.path(uri)?;
        let meta = self
            .store
            .head(&path)
            .await
            .map_err(|e| translate(uri, e))?;
        Ok(ObjectMeta {
            size: meta.size,
            etag: version_token(&meta),
            last_modified: meta.last_modified.into(),
        })
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = self.path(uri)?;
        retry::with_reissue(|| async {
            let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
            let meta = ObjectMeta {
                size: result.meta.size,
                etag: version_token(&result.meta),
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

    /// Single-RTT tail via GCS's native `Range: bytes=-len` suffix form; the
    /// response's `GetResult::meta.size` discloses the object size, so cold
    /// opens skip a separate HEAD (same shape as the S3 provider).
    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        if len == 0 {
            return Ok((Bytes::new(), self.head(uri).await?.size));
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
            let size = result.meta.size;
            let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
            Ok((bytes, size))
        })
        .await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let path = self.path(uri)?;
        // PutMode::Create maps to `x-goog-if-generation-match: 0` (native
        // create-if-absent). Re-issue only transient failures. Return the
        // new *generation* (r.version) — the CAS token callers chain forward
        // — not the HTTP etag (r.e_tag).
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
                    .map(|r| r.version)
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
            // Some(token) carries the GCS *generation* (see `version_token`);
            // it goes in `version`, not `e_tag`, or GCS errors MissingVersion.
            Some(expected) => PutOptions {
                mode: PutMode::Update(UpdateVersion {
                    e_tag: None,
                    version: Some(expected.to_string()),
                }),
                ..Default::default()
            },
        };
        // Return the new generation (r.version), same as put_atomic — this is
        // the token the WAL/manifest OCC loops chain into the next CAS.
        self.store
            .put_opts(&path, PutPayload::from_bytes(bytes), opts)
            .await
            .map(|r| r.version)
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
                    etag: version_token(&meta),
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
    //! Server-free unit tests: error translation, path/key building,
    //! constructors, and the generation-as-etag mapping. The trait impls
    //! (head/get/get_range/put_*/delete/list/tail) are exercised end-to-end
    //! against a real bucket in the gated `real_gcs` integration test.
    use chrono::DateTime;

    use super::*;

    fn test_provider() -> GcsStorageProvider {
        // Hermetic construction, no I/O: build points at a dead port but never
        // dials it, so path()/key()/bucket() are testable without network.
        let store = GoogleCloudStorageBuilder::new()
            .with_bucket_name("test-bucket")
            .with_base_url("http://127.0.0.1:1")
            .with_config(GoogleConfigKey::SkipSignature, "true")
            .build()
            .expect("build test store");
        GcsStorageProvider::from_object_store("test-bucket", store)
    }

    #[test]
    fn new_with_prefix_applies_a_bearer_token() {
        // A bearer token has no object_store config-key; the provider must apply
        // it via the credential provider and build, not reject it as unknown.
        let opts = StorageOptions::from([(
            GCS_BEARER_TOKEN_OPTION.to_string(),
            "ya29.a-test-access-token".to_string(),
        )]);
        let provider = GcsStorageProvider::new_with_prefix("test-bucket", "some/prefix", &opts)
            .expect("a bearer token builds a provider");
        assert_eq!(provider.bucket, "test-bucket");
    }

    #[test]
    fn new_with_prefix_pairs_a_bearer_token_with_config_keys() {
        // The bearer token is consumed; remaining `google_*` options still fold
        // on as config keys (here a base URL), so the two paths coexist.
        let opts = StorageOptions::from([
            (
                GCS_BEARER_TOKEN_OPTION.to_string(),
                "ya29.a-test-access-token".to_string(),
            ),
            (
                "google_base_url".to_string(),
                "http://127.0.0.1:1".to_string(),
            ),
        ]);
        let provider = GcsStorageProvider::new_with_prefix("test-bucket", "", &opts)
            .expect("bearer token and config keys coexist");
        assert_eq!(provider.bucket, "test-bucket");
    }

    #[test]
    fn new_with_prefix_still_rejects_an_unknown_config_key() {
        // Intercepting the bearer token must not weaken the loud-failure contract:
        // an unrelated unknown key is still rejected, not silently dropped.
        let opts = StorageOptions::from([("google_not_a_real_key".to_string(), "x".to_string())]);
        let err = GcsStorageProvider::new_with_prefix("test-bucket", "", &opts)
            .expect_err("an unknown config key is rejected");
        assert!(matches!(err, StorageError::Permanent { .. }));
    }

    #[test]
    fn translate_not_found_to_typed_variant() {
        let err = translate(
            "some/key",
            ObjError::NotFound {
                path: "some/key".into(),
                source: "raw".into(),
            },
        );
        assert!(matches!(err, StorageError::NotFound { uri } if uri == "some/key"));
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
                store: "GCS",
                source: "boom".into(),
            },
        );
        assert!(matches!(err, StorageError::TransientExhausted { uri, .. } if uri == "k"));
    }

    #[test]
    fn translate_other_variant_to_permanent() {
        let err = translate(
            "k",
            ObjError::UnknownConfigurationKey {
                store: "GCS",
                key: "foo".into(),
            },
        );
        assert!(matches!(err, StorageError::Permanent { uri, .. } if uri == "k"));
    }

    #[test]
    fn version_token_is_generation_never_etag() {
        // GCS CAS keys on the generation. The HTTP etag must NEVER stand in
        // for it — coercing an etag into the generation slot silently breaks
        // conditional writes, so absent generation => None, not the etag.
        let meta = OsMeta {
            location: "k".into(),
            last_modified: DateTime::from_timestamp(0, 0).expect("epoch"),
            size: 3,
            e_tag: Some("http-etag".into()),
            version: Some("42".into()),
        };
        assert_eq!(version_token(&meta).as_deref(), Some("42"));

        let meta_no_gen = OsMeta {
            version: None,
            ..meta
        };
        assert_eq!(
            version_token(&meta_no_gen),
            None,
            "no generation must yield None, never a fallback to the HTTP etag"
        );
    }

    #[test]
    fn normalize_prefix_trims_surrounding_slashes() {
        assert_eq!(normalize_prefix("/tbl/"), "tbl");
        assert_eq!(normalize_prefix("///a/b///"), "a/b");
        assert_eq!(normalize_prefix(""), "");
    }

    #[test]
    fn key_without_prefix_strips_leading_slash() {
        let p = test_provider();
        assert_eq!(p.prefix(), "");
        assert_eq!(p.key("/foo/bar"), "foo/bar");
        assert_eq!(p.key("foo/bar"), "foo/bar");
    }

    #[test]
    fn key_with_prefix_prepends_and_strips_leading_slash() {
        let mut p = test_provider();
        p.prefix = "tbl".into();
        assert_eq!(p.key("data/seg-1"), "tbl/data/seg-1");
        assert_eq!(p.key("/data/seg-1"), "tbl/data/seg-1");
    }

    #[test]
    fn path_parses_nested_uri() {
        let p = test_provider();
        assert_eq!(
            p.path("manifest-lists/list-000042.json")
                .expect("parse")
                .to_string(),
            "manifest-lists/list-000042.json"
        );
    }

    #[test]
    fn rejects_cross_backend_aws_key() {
        let opts = StorageOptions::from([("aws_region".to_string(), "us-east-1".to_string())]);
        assert!(GcsStorageProvider::new_with_prefix("b", "", &opts).is_err());
    }

    #[test]
    fn from_object_store_preserves_bucket() {
        let store = GoogleCloudStorageBuilder::new()
            .with_bucket_name("hatch-bucket")
            .with_base_url("http://127.0.0.1:1")
            .with_config(GoogleConfigKey::SkipSignature, "true")
            .build()
            .expect("build GoogleCloudStorage");
        assert_eq!(
            GcsStorageProvider::from_object_store("hatch-bucket", store).bucket(),
            "hatch-bucket"
        );
    }

    #[test]
    fn debug_impl_does_not_panic() {
        assert!(format!("{:?}", test_provider()).contains("GcsStorageProvider"));
    }

    #[test]
    fn object_store_handle_returns_path_under_prefix() {
        let mut p = test_provider();
        p.prefix = "tbl".into();
        let (_, path) = p
            .object_store_handle("data/seg-1")
            .expect("handle for valid uri");
        assert_eq!(path.to_string(), "tbl/data/seg-1");
    }
}
