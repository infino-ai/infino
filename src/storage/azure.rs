// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Azure Blob-backed [`StorageProvider`].
//!
//! Wraps `object_store::azure::MicrosoftAzure` so the supertable
//! exercises the same code paths on Azure as on LocalFS and S3.
//! Azure's container is the bucket-equivalent; conditional writes
//! (`PutMode::Create` / `PutMode::Update`) are native, so no
//! builder flag is needed to enable them.

use std::{fmt, ops::Range, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{
    ClientOptions, Error as ObjError, GetOptions, MultipartUpload, ObjectStore, ObjectStoreExt,
    PutMode, PutOptions, PutPayload, UpdateVersion,
    azure::{AzureConfigKey, MicrosoftAzure, MicrosoftAzureBuilder},
    client::{HttpClient, HttpConnector, HttpRequest, HttpRequestBody, ReqwestConnector},
    path::Path as ObjPath,
};

use super::{
    ObjectMeta, StorageError, StorageOptions, StorageProvider, child_prefix_names, counting,
    etag_meta, io_counters, logical_list_key, options::apply, retry,
};
use crate::{runtime_metrics::io::UsageMeter, utils::trace::record};

/// Azure Blob-backed `StorageProvider`. Cheap to clone; the inner
/// `MicrosoftAzure` shares its HTTP client across clones.
#[derive(Debug)]
pub struct AzureStorageProvider {
    container: String,
    prefix: String,
    store: Arc<MicrosoftAzure>,
    /// How a delete is issued when the credentials are a SAS; `None` deletes through `store`.
    sas_delete: Option<SasDelete>,
    meter: Arc<UsageMeter>,
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
        let sas_delete = SasDelete::for_builder(&builder, &container, &uri)?;
        let store = builder.build().map_err(|e| StorageError::Permanent {
            uri,
            source: Box::new(e),
        })?;
        Ok(Self {
            container,
            prefix: normalize_prefix(prefix),
            store: Arc::new(store),
            sas_delete,
            meter: UsageMeter::process_default(),
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
            sas_delete: None,
            meter: UsageMeter::process_default(),
        })
    }

    /// Wrap an already-constructed `MicrosoftAzure` — for callers
    /// that want full control over the `MicrosoftAzureBuilder`.
    pub fn from_object_store(container: impl Into<String>, store: MicrosoftAzure) -> Self {
        Self {
            container: container.into(),
            prefix: String::new(),
            store: Arc::new(store),
            sas_delete: None,
            meter: UsageMeter::process_default(),
        }
    }

    /// One DELETE of `path`: by its own request under a SAS ([`SasDelete`]), else through the
    /// store.
    async fn delete_blob(&self, path: &ObjPath) -> Result<(), ObjError> {
        match &self.sas_delete {
            Some(sas) => sas.delete(path).await,
            None => self.store.delete(path).await,
        }
    }

    /// Replace the usage meter (connection-scoped ledger).
    pub fn with_usage_meter(mut self, meter: Arc<UsageMeter>) -> Self {
        self.meter = meter;
        self
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
///
/// This must stay strictly *below* the retry budget
/// ([`retry::RETRY_TIMEOUT`]): the timeout bounds one request, the budget
/// bounds the whole retry loop. If the two were equal, a single stalled
/// request would burn the entire budget and the retry loop would give up
/// after one attempt (surfacing as a fatal `TransientExhausted` /
/// "error sending request" on a large commit) instead of re-dialing on a
/// fresh connection. At 60s against the 300s budget the loop still gets
/// ~5 attempts while the worst-case total hang stays at 300s (not
/// doubled). 60s comfortably covers each 8 MiB multipart part and a
/// typical single-shot `put_atomic` (the ~64 MiB default superfile split)
/// down to ~1 MB/s; the only edge case — a ~100 MiB single-shot on a
/// sub-1.7 MB/s link — is removed by the deferred multipart-threshold
/// reduction (see the request's future-work notes). A stalled request now
/// fails in 60s and is retried rather than being fatal.
const AZURE_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// HTTP client options: deep warm idle pool + bounded connect/request.
fn tuned_client_options() -> ClientOptions {
    ClientOptions::new()
        .with_pool_max_idle_per_host(AZURE_POOL_MAX_IDLE_PER_HOST)
        .with_pool_idle_timeout(AZURE_POOL_IDLE_TIMEOUT)
        .with_connect_timeout(AZURE_CONNECT_TIMEOUT)
        .with_timeout(AZURE_REQUEST_TIMEOUT)
}

/// The store name errors from [`SasDelete`] carry, as `object_store`'s own Azure errors do.
const AZURE_STORE: &str = "MicrosoftAzure";

/// The scheme of an endpoint served without TLS (an emulator), which the client must be told to
/// allow.
const PLAIN_HTTP: &str = "http://";

/// The method of a Delete Blob request.
const DELETE_METHOD: &str = "DELETE";

/// The status of a refused request: its credentials are missing or not valid.
const STATUS_UNAUTHENTICATED: u16 = 401;

/// The status of a refused request: its credentials do not allow the operation.
const STATUS_FORBIDDEN: u16 = 403;

/// The status of a request for a blob that does not exist.
const STATUS_NOT_FOUND: u16 = 404;

/// The status of a request the service gave up waiting for.
const STATUS_REQUEST_TIMEOUT: u16 = 408;

/// The status of a request refused for coming too fast.
const STATUS_TOO_MANY_REQUESTS: u16 = 429;

/// The first status that reports a fault of the service's own.
const STATUS_SERVER_ERROR_MIN: u16 = 500;

/// Most bytes of a refusal's body kept in the error: enough for the service's error code and
/// detail.
const ERROR_BODY_MAX: usize = 512;

/// Deletes blobs one request each, signed by a SAS.
///
/// `object_store` issues every Azure delete, a single one included, as a Blob Batch request: a
/// POST to the container. A SAS signs one resource. An account or container SAS signs the
/// container, so the batch is accepted. A SAS for a directory (`sr=d`) or a blob does not, and the
/// service refuses the batch with `403 AuthenticationFailed` ("Signed Directory Depth Invalid" for
/// a directory), whatever the batch holds: under such a SAS no delete ever succeeds. A Delete Blob
/// request names the blob itself, which every SAS that grants delete on it signs, so under any SAS
/// a delete is sent that way. It is one request per blob either way.
struct SasDelete {
    http: HttpClient,
    /// The container's URL, with no trailing slash and no query.
    container_url: String,
    /// The SAS as a query string, with no leading `?`. A secret: never logged or put in an error.
    sas: String,
}

impl fmt::Debug for SasDelete {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SasDelete")
            .field("container_url", &self.container_url)
            .finish_non_exhaustive()
    }
}

impl SasDelete {
    /// The per-blob delete for a store `builder` makes of `container`, or `None` when its
    /// credentials are not a SAS (or it names no account to address), and deletes go through the
    /// store.
    fn for_builder(
        builder: &MicrosoftAzureBuilder,
        container: &str,
        uri: &str,
    ) -> Result<Option<Self>, StorageError> {
        let Some(sas) = builder.get_config_value(&AzureConfigKey::SasKey) else {
            return Ok(None);
        };
        let endpoint = match builder.get_config_value(&AzureConfigKey::Endpoint) {
            Some(endpoint) => endpoint,
            None => match builder.get_config_value(&AzureConfigKey::AccountName) {
                Some(account) => format!("https://{account}.blob.core.windows.net"),
                None => return Ok(None),
            },
        };
        let options = tuned_client_options().with_allow_http(endpoint.starts_with(PLAIN_HTTP));
        let http =
            ReqwestConnector::default()
                .connect(&options)
                .map_err(|e| StorageError::Permanent {
                    uri: uri.to_string(),
                    source: Box::new(e),
                })?;
        Ok(Some(Self {
            http,
            container_url: format!("{}/{container}", endpoint.trim_end_matches('/')),
            sas: sas.trim_start_matches('?').to_string(),
        }))
    }

    /// The URL of a Delete Blob request for `path`. It carries the SAS: never log it.
    fn blob_url(&self, path: &ObjPath) -> String {
        let blob = path
            .parts()
            .map(|part| encode_segment(part.as_ref()))
            .collect::<Vec<_>>()
            .join("/");
        format!("{}/{blob}?{}", self.container_url, self.sas)
    }

    /// One Delete Blob request for `path`, its answer in `object_store`'s terms so the provider
    /// settles and translates it as it does the store's own: a missing blob is `NotFound`, refused
    /// credentials are `Unauthenticated` or `PermissionDenied`, and a timeout, a dropped connection
    /// or a fault of the service is `Generic`, which the provider re-issues.
    async fn delete(&self, path: &ObjPath) -> Result<(), ObjError> {
        let unsendable = || ObjError::NotSupported {
            source: format!("no Delete Blob request can be made for {path}").into(),
        };
        let mut request = HttpRequest::new(HttpRequestBody::empty());
        *request.method_mut() = DELETE_METHOD.parse().map_err(|_| unsendable())?;
        // The parse error is dropped, not kept as the source: the URL holds the SAS.
        *request.uri_mut() = self.blob_url(path).parse().map_err(|_| unsendable())?;
        let response = self
            .http
            .execute(request)
            .await
            .map_err(|e| ObjError::Generic {
                store: AZURE_STORE,
                source: Box::new(e),
            })?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let code = status.as_u16();
        // The body is the service's error document: its code and detail, and nothing of the SAS.
        let body = response.into_body().bytes().await.unwrap_or_default();
        let detail = String::from_utf8_lossy(&body[..body.len().min(ERROR_BODY_MAX)]).into_owned();
        let source = format!("Delete Blob answered {code}: {detail}").into();
        let path = path.to_string();
        Err(match code {
            STATUS_NOT_FOUND => ObjError::NotFound { path, source },
            STATUS_UNAUTHENTICATED => ObjError::Unauthenticated { path, source },
            STATUS_FORBIDDEN => ObjError::PermissionDenied { path, source },
            STATUS_REQUEST_TIMEOUT | STATUS_TOO_MANY_REQUESTS => ObjError::Generic {
                store: AZURE_STORE,
                source,
            },
            code if code >= STATUS_SERVER_ERROR_MIN => ObjError::Generic {
                store: AZURE_STORE,
                source,
            },
            _ => ObjError::NotSupported { source },
        })
    }
}

/// `segment` of a blob's path as it goes in a URL: every byte but an unreserved one
/// (`A-Z a-z 0-9 - . _ ~`) percent-encoded.
fn encode_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
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
        // Refused credentials, kept apart from `Permanent`: the URI is
        // fine and the same call with valid credentials can succeed.
        ObjError::PermissionDenied { .. } | ObjError::Unauthenticated { .. } => {
            StorageError::PermissionDenied { uri: uri.into() }
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
        self.meter.record_head();
        Ok(etag_meta(meta))
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            skip_all,
            fields(uri = uri, len = tracing::field::Empty, hedged = tracing::field::Empty)
        )
    )]
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = self.path(uri)?;
        let tl = io_counters::timeline_start();
        // etag and bytes are atomically paired in the same response, so
        // no follow-up HEAD is needed.
        let out = retry::read_body(uri, retry::Hedging::reads(&self.meter), || async {
            let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
            Ok(retry::Response::new(result, move |e| translate(uri, e)))
        })
        .await
        .map(|(bytes, meta)| (bytes, etag_meta(meta)));
        if let Ok((b, _)) = &out {
            self.meter.record_get(uri, None, b.len() as u64);
            record("len", b.len() as u64);
            io_counters::timeline_record("get", uri, 0, b.len() as u64, tl);
        }
        out
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            skip_all,
            fields(uri = uri, len = tracing::field::Empty, hedged = tracing::field::Empty)
        )
    )]
    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        let path = self.path(uri)?;
        // Native `If-None-Match`: an unchanged blob comes back as a
        // bodyless 304 instead of a full read. Both arms are still a
        // billable GET (0 body bytes on 304).
        let out = retry::read_whole(uri, retry::Hedging::reads(&self.meter), || async {
            let options = GetOptions {
                if_none_match: Some(etag.to_string()),
                ..GetOptions::default()
            };
            match self.store.get_opts(&path, options).await {
                Ok(result) => Ok(Some(retry::Response::new(result, move |e| {
                    translate(uri, e)
                }))),
                Err(ObjError::NotModified { .. }) => Ok(None),
                Err(e) => Err(translate(uri, e)),
            }
        })
        .await
        .map(|read| read.map(|(bytes, meta)| (bytes, etag_meta(meta))));
        match &out {
            Ok(Some((b, _))) => {
                self.meter.record_get(uri, None, b.len() as u64);
                record("len", b.len() as u64);
            }
            Ok(None) => {
                self.meter.record_get(uri, None, 0);
                record("len", 0);
            }
            Err(_) => {}
        }
        out
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            skip_all,
            fields(uri = uri, len = range.end - range.start, hedged = tracing::field::Empty)
        )
    )]
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = self.path(uri)?;
        let requested = (range.start, range.end);
        let off = range.start;
        let tl = io_counters::timeline_start();
        let out = retry::read_range(uri, range, retry::Hedging::reads(&self.meter), |r| async {
            let result = self
                .store
                .get_opts(&path, GetOptions::new().with_range(Some(r)))
                .await
                .map_err(|e| translate(uri, e))?;
            Ok(retry::Response::new(result, move |e| translate(uri, e)))
        })
        .await;
        if let Ok(b) = &out {
            self.meter.record_get(uri, Some(requested), b.len() as u64);
            io_counters::timeline_record("get_range", uri, off, b.len() as u64, tl);
        }
        out
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
        let n = bytes.len() as u64;
        // Re-issue transient failures like the read paths. A create the
        // store accepted can still come back failed, so every issue carries
        // this call's id and a `PreconditionFailed` is settled with one HEAD
        // (see `create_with_reissue`).
        let id = retry::CreateId::new();
        let out = retry::create_with_reissue(
            uri,
            &id,
            || {
                let bytes = bytes.clone();
                async {
                    self.store
                        .put_opts(&path, PutPayload::from_bytes(bytes), id.put_options())
                        .await
                        .map(|r| r.e_tag)
                        .map_err(|e| translate(uri, e))
                }
            },
            || async {
                let head = self
                    .store
                    .get_opts(&path, retry::head_options())
                    .await
                    .map_err(|e| translate(uri, e))?;
                self.meter.record_head();
                Ok((head.attributes, head.meta.e_tag))
            },
        )
        .await;
        if out.is_ok() {
            self.meter.record_put(n);
        }
        out
    }

    async fn put_overwrite(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        let path = self.path(uri)?;
        let n = bytes.len() as u64;
        // Overwrite is idempotent, so a transient failure re-issues like the
        // create-only path; there is no precondition to lose.
        let out = retry::with_reissue(|| {
            let bytes = bytes.clone();
            async {
                let opts = PutOptions {
                    mode: PutMode::Overwrite,
                    ..Default::default()
                };
                self.store
                    .put_opts(&path, PutPayload::from_bytes(bytes), opts)
                    .await
                    .map(|_| ())
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
        retry::delete_with_reissue(
            || async { self.delete_blob(&path).await },
            |e| translate(uri, e),
        )
        .await?;
        self.meter.record_delete();
        Ok(())
    }

    async fn delete_once(&self, uri: &str) -> Result<(), StorageError> {
        let path = self.path(uri)?;
        retry::delete_once(async { self.delete_blob(&path).await }, |e| {
            translate(uri, e)
        })
        .await?;
        self.meter.record_delete();
        Ok(())
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
            out.push((logical_list_key(&self.prefix, &location), etag_meta(meta)));
        }
        Ok(out)
    }

    async fn list_child_prefixes(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let path = self.path(prefix)?;
        self.meter.record_list();
        child_prefix_names(self.store.as_ref(), &path)
            .await
            .map_err(|e| translate(prefix, e))
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
    //! Unit tests for the parts that don't need a live backend:
    //! error translation, path parsing, the emulator constructor,
    //! and `from_object_store`. The trait impls are
    //! exercised end-to-end against Azurite in the gated
    //! `supertable_smoke_via_azure_wire_protocol` integration test.
    use std::{
        io::{BufRead, BufReader, Read as _, Write},
        net::TcpListener,
        sync::Mutex,
        thread,
    };

    use super::*;
    use crate::storage::stalling_endpoint::{
        Endpoint, Read, Stall, assert_stalled_read_recovers, stalling_endpoint,
    };

    /// The account and container the stalled-read tests read from (the stalling endpoint serves
    /// any path).
    const STALL_ACCOUNT: &str = "acct";
    const STALL_CONTAINER: &str = "container";

    /// The container and prefix the SAS delete tests delete under.
    const SAS_CONTAINER: &str = "data";
    const SAS_PREFIX: &str = "tenant/db";

    /// A SAS for a directory, as a query string. `sig` holds an encoded `+` and `/`, which must
    /// reach the service as they are.
    const DIRECTORY_SAS: &str = "sv=2022-11-02&sr=d&sdd=2&sp=racwdl&sig=qfK%2BlW%2FsecretSig%3D";

    /// The signature inside [`DIRECTORY_SAS`], which no error may show.
    const SAS_SIGNATURE: &str = "secretSig";

    /// The service's answer to a batch signed by a directory SAS, and to any refused request here.
    const REFUSAL_BODY: &str = "<Error><Code>AuthenticationFailed</Code>\
        <AuthenticationErrorDetail>Signed Directory Depth Invalid</AuthenticationErrorDetail></Error>";

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
    fn translate_refused_credentials_to_permission_denied() {
        // 403 and 401 both mean the credentials were refused: kept apart from
        // `Permanent` so a caller can supply fresh ones and reissue instead of
        // treating the URI as broken.
        for e in [
            ObjError::PermissionDenied {
                path: "k".into(),
                source: "forbidden".into(),
            },
            ObjError::Unauthenticated {
                path: "k".into(),
                source: "expired".into(),
            },
        ] {
            let err = translate("k", e);
            assert!(matches!(err, StorageError::PermissionDenied { uri } if uri == "k"));
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

    // ---- timeout invariant ---------------------------------------------

    #[test]
    fn request_timeout_stays_below_retry_budget() {
        // The per-request timeout must be strictly less than the retry
        // budget, or a single stalled request burns the whole budget and no
        // retry fires. (Regression: the two were both 300s, so a stalled
        // block PUT gave up after one attempt.)
        assert!(
            AZURE_REQUEST_TIMEOUT < crate::storage::retry::RETRY_TIMEOUT,
            "request timeout ({AZURE_REQUEST_TIMEOUT:?}) must be below the retry budget \
             ({:?}) so a stalled request is retried, not fatal",
            crate::storage::retry::RETRY_TIMEOUT,
        );
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
            .path("manifest/manifest-000042.json")
            .expect("parse");
        assert_eq!(p.to_string(), "manifest/manifest-000042.json");
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
    // exists the way RustFS does for S3), so it cannot be stood up from
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

    // ---- deletes under a SAS ---------------------------------------------

    /// A plain-HTTP endpoint that answers every request with one status, and keeps each request's
    /// first line (`METHOD target HTTP/1.1`) in the order they came.
    struct RecordingEndpoint {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingEndpoint {
        /// Start one on a free port that answers `status_line` (as `202 Accepted`) with `body`.
        fn answering(status_line: &'static str, body: &'static str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let seen = Arc::clone(&requests);
            thread::spawn(move || {
                for conn in listener.incoming() {
                    let Ok(mut conn) = conn else { return };
                    let seen = Arc::clone(&seen);
                    thread::spawn(move || {
                        let mut reader = BufReader::new(conn.try_clone().expect("clone"));
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                                return;
                            }
                            let mut content_length = 0usize;
                            loop {
                                let mut header = String::new();
                                if reader.read_line(&mut header).unwrap_or(0) == 0 {
                                    return;
                                }
                                if header == "\r\n" {
                                    break;
                                }
                                let header = header.to_ascii_lowercase();
                                if let Some(len) = header.strip_prefix("content-length:") {
                                    content_length = len.trim().parse().expect("length");
                                }
                            }
                            // A batch request carries a body, which must be read off before the
                            // next request's line.
                            let mut sent = vec![0u8; content_length];
                            if reader.read_exact(&mut sent).is_err() {
                                return;
                            }
                            seen.lock().expect("requests").push(line.trim().to_string());
                            let answer = format!(
                                "HTTP/1.1 {status_line}\r\nContent-Length: {}\r\n\r\n{body}",
                                body.len()
                            );
                            if conn.write_all(answer.as_bytes()).is_err() {
                                return;
                            }
                        }
                    });
                }
            });
            Self {
                url: format!("http://{addr}"),
                requests,
            }
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().expect("requests").clone()
        }
    }

    /// A provider under [`SAS_PREFIX`] of [`SAS_CONTAINER`] whose credentials are
    /// [`DIRECTORY_SAS`] and whose service is `endpoint`.
    fn sas_provider(endpoint: &RecordingEndpoint) -> AzureStorageProvider {
        let opts = StorageOptions::from([
            ("azure_storage_account_name".to_string(), "acct".to_string()),
            ("azure_storage_endpoint".to_string(), endpoint.url.clone()),
            (
                "azure_storage_sas_token".to_string(),
                DIRECTORY_SAS.to_string(),
            ),
        ]);
        AzureStorageProvider::new_with_prefix(SAS_CONTAINER, SAS_PREFIX, &opts).expect("build")
    }

    /// Under a SAS a delete is one Delete Blob request for the blob itself, with the SAS as its
    /// query, untouched. Before, it was a Blob Batch POST to the container, which a SAS for a
    /// directory cannot sign: the service refused every delete.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_delete_under_a_sas_names_the_blob_itself() {
        let endpoint = RecordingEndpoint::answering("202 Accepted", "");
        let provider = sas_provider(&endpoint);
        provider
            .delete("wal/mutations/0001.arrow")
            .await
            .expect("the blob's own URL is one the SAS signs");
        provider
            .delete_once("wal/mutations/0001.json")
            .await
            .expect("delete_once goes the same way");
        assert_eq!(
            endpoint.requests(),
            [
                format!(
                    "DELETE /{SAS_CONTAINER}/{SAS_PREFIX}/wal/mutations/0001.arrow?{DIRECTORY_SAS} HTTP/1.1"
                ),
                format!(
                    "DELETE /{SAS_CONTAINER}/{SAS_PREFIX}/wal/mutations/0001.json?{DIRECTORY_SAS} HTTP/1.1"
                ),
            ]
        );
    }

    /// A blob that is already gone is deleted: a re-issued delete, or a sweep run twice, converges.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_delete_under_a_sas_of_a_missing_blob_succeeds() {
        let endpoint = RecordingEndpoint::answering("404 Not Found", "<Error/>");
        sas_provider(&endpoint)
            .delete("data/gone.parquet")
            .await
            .expect("a missing blob counts as deleted");
        assert_eq!(endpoint.requests().len(), 1);
    }

    /// A refused delete is `PermissionDenied`, at once and not re-issued, and says what the
    /// service said without showing the SAS.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refused_delete_under_a_sas_is_permission_denied_and_hides_the_sas() {
        let endpoint = RecordingEndpoint::answering("403 Forbidden", REFUSAL_BODY);
        let provider = sas_provider(&endpoint);
        let err = provider
            .delete("data/seg.parquet")
            .await
            .expect_err("refused");
        assert!(
            matches!(&err, StorageError::PermissionDenied { uri } if uri == "data/seg.parquet")
        );
        assert_eq!(endpoint.requests().len(), 1);

        let path = provider.path("data/seg.parquet").expect("path");
        let per_blob = provider
            .sas_delete
            .as_ref()
            .expect("a SAS provider deletes per blob");
        let raw = per_blob.delete(&path).await.expect_err("refused");
        let shown = format!("{raw} {raw:?} {per_blob:?}");
        assert!(shown.contains("Signed Directory Depth Invalid"), "{shown}");
        assert!(!shown.contains(SAS_SIGNATURE), "{shown}");
    }

    /// A fault of the service's own is transient, as the store's is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_service_fault_on_a_delete_under_a_sas_is_transient() {
        let endpoint = RecordingEndpoint::answering("503 Service Unavailable", "<Error/>");
        let provider = sas_provider(&endpoint);
        let err = provider
            .delete_once("data/seg.parquet")
            .await
            .expect_err("unavailable");
        assert!(matches!(err, StorageError::TransientExhausted { .. }));
    }

    /// Credentials that are not a SAS sign a batch, so those deletes stay with the store.
    #[test]
    fn only_a_sas_deletes_per_blob() {
        let key = StorageOptions::from([
            ("azure_storage_account_name".to_string(), "acct".to_string()),
            ("azure_storage_account_key".to_string(), "a2V5".to_string()),
        ]);
        let provider = AzureStorageProvider::new_with_prefix("c", "p", &key).expect("build");
        assert!(provider.sas_delete.is_none());
        assert!(test_provider().sas_delete.is_none());

        let sas = StorageOptions::from([
            ("azure_storage_account_name".to_string(), "acct".to_string()),
            (
                "azure_storage_sas_key".to_string(),
                format!("?{DIRECTORY_SAS}"),
            ),
        ]);
        let provider = AzureStorageProvider::new_with_prefix("c", "p", &sas).expect("build");
        let per_blob = provider
            .sas_delete
            .as_ref()
            .expect("a SAS deletes per blob");
        let path = provider.path("a b/ü.parquet").expect("path");
        assert_eq!(
            per_blob.blob_url(&path),
            format!("https://acct.blob.core.windows.net/c/p/a%20b/%C3%BC.parquet?{DIRECTORY_SAS}")
        );
    }

    // ---- stalled reads, through the real client ---------------------------

    /// An Azure provider whose store is `endpoint`, its usage counted on `meter`.
    fn stalling_provider(endpoint: &Endpoint, meter: &Arc<UsageMeter>) -> AzureStorageProvider {
        let store = MicrosoftAzureBuilder::new()
            .with_account(STALL_ACCOUNT)
            .with_container_name(STALL_CONTAINER)
            .with_endpoint(endpoint.url.clone())
            .with_skip_signature(true)
            .with_client_options(tuned_client_options().with_allow_http(true))
            .with_retry(retry::config())
            .build()
            .expect("store");
        AzureStorageProvider::from_object_store(STALL_CONTAINER, store)
            .with_usage_meter(Arc::clone(meter))
    }

    /// Through the real Azure client: a range GET whose headers never arrive gets a second request
    /// after its delay, so the read returns in seconds instead of waiting out the 60 s request
    /// timeout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_range_get_with_stalled_headers_is_answered_by_its_hedge() {
        let endpoint = stalling_endpoint(Stall::BeforeHeaders);
        let meter = UsageMeter::new();
        let provider = stalling_provider(&endpoint, &meter);
        assert_stalled_read_recovers(
            &provider,
            &meter,
            &endpoint,
            Stall::BeforeHeaders,
            Read::Range,
        )
        .await;
    }

    /// Through the real Azure client: a range GET whose body stops partway is dropped after its
    /// idle timeout and only its missing bytes are asked for again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_range_get_with_a_stalled_body_asks_again_for_its_missing_bytes() {
        let endpoint = stalling_endpoint(Stall::MidBody);
        let meter = UsageMeter::new();
        let provider = stalling_provider(&endpoint, &meter);
        assert_stalled_read_recovers(&provider, &meter, &endpoint, Stall::MidBody, Read::Range)
            .await;
    }

    /// Through the real Azure client: a whole-object GET whose body stops partway is dropped after
    /// its idle timeout and sent again whole.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_get_with_a_stalled_body_is_sent_again() {
        let endpoint = stalling_endpoint(Stall::MidBody);
        let meter = UsageMeter::new();
        let provider = stalling_provider(&endpoint, &meter);
        assert_stalled_read_recovers(&provider, &meter, &endpoint, Stall::MidBody, Read::Whole)
            .await;
    }
}
