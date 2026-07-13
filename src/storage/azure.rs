// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Azure Blob-backed [`StorageProvider`].
//!
//! Wraps `object_store::azure::MicrosoftAzure` so the supertable
//! exercises the same code paths on Azure as on LocalFS and S3.
//! Azure's container is the bucket-equivalent; conditional writes
//! (`PutMode::Create` / `PutMode::Update`) are native, so no
//! builder flag is needed to enable them.

use std::{ops::Range, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{
    ClientOptions, Error as ObjError, GetOptions, MultipartUpload, ObjectStore, ObjectStoreExt,
    PutMode, PutOptions, PutPayload, UpdateVersion,
    azure::{AzureConfigKey, MicrosoftAzure, MicrosoftAzureBuilder},
    path::Path as ObjPath,
};

use super::{ObjectMeta, StorageError, StorageOptions, StorageProvider, options::apply, retry};

/// Azure Blob-backed `StorageProvider`. Cheap to clone; the inner
/// `MicrosoftAzure` shares its HTTP client across clones.
#[derive(Debug)]
pub struct AzureStorageProvider {
    container: String,
    prefix: String,
    store: Arc<MicrosoftAzure>,
}

impl AzureStorageProvider {
    /// Azure provider for `container` with no explicit options —
    /// credentials resolve through object_store's ambient chain (managed
    /// identity). Infino never reads Azure credentials from the process
    /// environment; pass them through [`Self::new_with_prefix`] otherwise.
    pub fn new(container: impl Into<String>) -> Result<Self, StorageError> {
        Self::new_with_prefix(container, "", &StorageOptions::new())
    }

    /// Azure provider scoped to `prefix` inside `container`, configured
    /// from `opts` (account/key, keyed by object_store's `azure_*`
    /// strings). The prefix isolates each table under
    /// `azure://container/prefix/`.
    pub fn new_with_prefix(
        container: impl Into<String>,
        prefix: impl Into<String>,
        opts: &StorageOptions,
    ) -> Result<Self, StorageError> {
        let container = container.into();
        let uri = format!("azure://{container}");
        let builder = MicrosoftAzureBuilder::new()
            .with_container_name(&container)
            .with_client_options(tuned_client_options())
            .with_retry(retry::config());
        let builder = apply::<AzureConfigKey, _>(builder, opts, &uri, |b, key, value| {
            b.with_config(key, value)
        })?;
        let store = builder.build().map_err(|e| StorageError::Permanent {
            uri,
            source: Box::new(e),
        })?;
        Ok(Self {
            container,
            prefix: normalize_prefix(prefix),
            store: Arc::new(store),
        })
    }

    /// Construct against the Azurite emulator. `with_use_emulator`
    /// injects the well-known `devstoreaccount1` credentials, the
    /// `http://127.0.0.1:10000` endpoint, and plain-HTTP support —
    /// so no credentials are passed and `tuned_client_options` is
    /// deliberately not applied (it would override the emulator's
    /// `allow_http`).
    pub fn new_with_emulator(container: impl Into<String>) -> Result<Self, StorageError> {
        let container = container.into();
        let store = MicrosoftAzureBuilder::new()
            .with_use_emulator(true)
            .with_container_name(&container)
            .build()
            .map_err(|e| StorageError::Permanent {
                uri: format!("azure://{container} @ emulator"),
                source: Box::new(e),
            })?;
        Ok(Self {
            container,
            prefix: String::new(),
            store: Arc::new(store),
        })
    }

    /// Wrap an already-constructed `MicrosoftAzure` — for callers
    /// that want full control over the `MicrosoftAzureBuilder`.
    pub fn from_object_store(container: impl Into<String>, store: MicrosoftAzure) -> Self {
        Self {
            container: container.into(),
            prefix: String::new(),
            store: Arc::new(store),
        }
    }

    /// Container this provider is scoped to.
    pub fn container(&self) -> &str {
        &self.container
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

/// Warm idle connections per host, so a wide range-GET fan-out reuses
/// TLS sessions instead of re-handshaking on the cold tail.
const AZURE_POOL_MAX_IDLE_PER_HOST: usize = 1024;

/// Idle-connection keep-alive, below Azure's server-side close window.
const AZURE_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Connect-phase timeout, so one slow SYN/TLS can't dominate the p99.
const AZURE_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Whole-request timeout (incl. body). The 30s default is too tight for
/// a multi-MB superfile PUT on a modest uplink — it aborts mid-upload.
const AZURE_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// HTTP client options: deep warm idle pool + bounded connect/request.
fn tuned_client_options() -> ClientOptions {
    ClientOptions::new()
        .with_pool_max_idle_per_host(AZURE_POOL_MAX_IDLE_PER_HOST)
        .with_pool_idle_timeout(AZURE_POOL_IDLE_TIMEOUT)
        .with_connect_timeout(AZURE_CONNECT_TIMEOUT)
        .with_timeout(AZURE_REQUEST_TIMEOUT)
}

/// Translate an `object_store::Error` to our `StorageError`. Kept
/// per-backend (not shared) so each backend's mapping can diverge if
/// its error surface widens.
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
impl StorageProvider for AzureStorageProvider {
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

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        let path = self.path(uri)?;
        // Native `If-None-Match`: an unchanged blob comes back as a
        // bodyless 304 instead of a full read.
        retry::with_reissue(|| async {
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

    /// Tail fetch on Azure. Unlike S3, object_store's Azure backend
    /// rejects suffix ranges (`Range: bytes=-len` → "Operation not
    /// supported"), so the object size is resolved with a HEAD and the
    /// trailing `len` bytes are pulled with a bounded `get_range`. That
    /// is two RTTs rather than the single-RTT suffix form, but it is the
    /// only shape Azure accepts; the size still comes back for parity
    /// with the default impl. (Production supertable reads carry
    /// `total_size` in the manifest and never reach this path; the
    /// sizeless tail open is the standalone-superfile reader's cold open.)
    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let size = self.head(uri).await?.size;
        if len == 0 {
            return Ok((Bytes::new(), size));
        }
        let start = size.saturating_sub(len);
        let bytes = self.get_range(uri, start..size).await?;
        Ok((bytes, size))
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
            // Some(tag) == native conditional update; Azure enforces
            // the etag match atomically, returning 412 on conflict,
            // which `translate` maps to `PreconditionFailed`.
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
    //! Unit tests for the parts that don't need a live backend:
    //! error translation, path parsing, the emulator constructor,
    //! and `from_object_store`. The trait impls are
    //! exercised end-to-end against Azurite in the gated
    //! `supertable_smoke_via_azure_wire_protocol` integration test.
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
                store: "MicrosoftAzure",
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
        let err = translate(
            "k",
            ObjError::UnknownConfigurationKey {
                store: "MicrosoftAzure",
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
        let p = test_provider().path("foo/bar.txt").expect("parse");
        assert_eq!(p.to_string(), "foo/bar.txt");
    }

    #[test]
    fn path_parses_nested_uri() {
        let p = test_provider()
            .path("manifest-lists/list-000042.json")
            .expect("parse");
        assert_eq!(p.to_string(), "manifest-lists/list-000042.json");
    }

    #[test]
    fn path_applies_prefix() {
        let mut p = test_provider();
        p.prefix = "tbl".into();
        assert_eq!(p.key("data/seg-1"), "tbl/data/seg-1");
    }

    // ---- constructors --------------------------------------------------

    fn test_provider() -> AzureStorageProvider {
        // The emulator constructor builds without I/O, so it's a cheap
        // way to exercise `path()` / `key()` / Debug without a backend.
        AzureStorageProvider::new_with_emulator("test-container")
            .expect("construct emulator provider")
    }

    #[test]
    fn new_with_emulator_builds_and_exposes_container() {
        let p = AzureStorageProvider::new_with_emulator("emu-container")
            .expect("construct with emulator");
        assert_eq!(p.container(), "emu-container");
    }

    #[test]
    fn applies_account_options_and_exposes_container() {
        let opts = StorageOptions::from([
            ("azure_storage_account_name".to_string(), "acct".to_string()),
            ("azure_storage_account_key".to_string(), "a2V5".to_string()),
        ]);
        let p = AzureStorageProvider::new_with_prefix("c", "p", &opts).expect("build azure");
        assert_eq!(p.container(), "c");
        assert_eq!(p.prefix(), "p");
    }

    #[test]
    fn rejects_cross_backend_aws_key() {
        let opts = StorageOptions::from([("aws_region".to_string(), "us-east-1".to_string())]);
        assert!(AzureStorageProvider::new_with_prefix("c", "", &opts).is_err());
    }

    #[test]
    fn from_object_store_preserves_container() {
        let store = MicrosoftAzureBuilder::new()
            .with_endpoint("http://127.0.0.1:1".to_string())
            .with_container_name("hatch-container")
            .with_account("devstoreaccount1")
            .with_access_key("dGVzdC1rZXk=")
            .with_allow_http(true)
            .build()
            .expect("build MicrosoftAzure");
        let p = AzureStorageProvider::from_object_store("hatch-container", store);
        assert_eq!(p.container(), "hatch-container");
    }

    #[test]
    fn debug_impl_does_not_panic() {
        let p = test_provider();
        let s = format!("{p:?}");
        assert!(s.contains("AzureStorageProvider"));
    }

    // ---- pure helpers: prefix / key ------------------------------------
    //
    // The Azure round-trip paths (head / get / get_range / put_atomic /
    // put_if_match / delete / list_with_prefix / tail) are exercised
    // end-to-end against the Azurite emulator in the gated
    // `supertable_smoke_via_azure_wire_protocol` integration test. Azurite
    // is an out-of-process Docker emulator (no in-process server crate
    // exists the way `s3s-fs` does for S3), so it cannot be stood up from
    // a `#[cfg(test)]` unit test here; these unit tests therefore cover the
    // pure, server-free surface (constructors, `translate`, path/key
    // building) directly.

    #[test]
    fn normalize_prefix_trims_surrounding_slashes() {
        assert_eq!(normalize_prefix("/tbl/"), "tbl");
        assert_eq!(normalize_prefix("///a/b///"), "a/b");
        assert_eq!(normalize_prefix("plain"), "plain");
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
        assert_eq!(p.prefix(), "tbl");
        assert_eq!(p.key("data/seg-1"), "tbl/data/seg-1");
        assert_eq!(p.key("/data/seg-1"), "tbl/data/seg-1");
    }

    #[test]
    fn path_with_prefix_parses_under_prefix() {
        let mut p = test_provider();
        p.prefix = "tbl".into();
        let parsed = p.path("data/seg-1").expect("parse");
        assert_eq!(parsed.to_string(), "tbl/data/seg-1");
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
