// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Storage provider abstraction over object stores.
//!
//! Wraps the `object_store` crate with a narrower, supertable-
//! shaped interface exposing only the operations the supertable's
//! manifest + disk-cache layers consume:
//!
//! - `head` / `get` / `get_range` — read paths.
//! - `put_atomic` / `put_if_match` / `put_multipart` — write
//!   paths; `put_atomic` and `put_if_match` are the
//!   conditional-write primitives the manifest's OCC + the
//!   atomic-rename pointer commit ride on.
//! - `delete` — idempotent object removal.
//!
//! ## Retry contract
//!
//! Implementations inherit `object_store`'s internal bounded
//! retry of transient failures (5xx, connection-reset,
//! timeouts) under its `RetryConfig`. The `Result` returned by
//! a `StorageProvider` method therefore represents either a
//! *permanent* failure or a *transient failure that exhausted
//! the provider's retry budget*. Callers do **not** retry
//! transient errors themselves.
//!
//! The single exception is OCC on the manifest pointer:
//! [`StorageError::PreconditionFailed`] is a legitimate
//! contention signal. The supertable's commit loop catches it
//! specifically, re-reads the pointer to capture the winner's
//! state, and retries the commit on top of it.

use std::{fmt, ops::Range, sync::Arc, time::SystemTime};

use async_trait::async_trait;
use bytes::Bytes;
use thiserror::Error;

pub mod azure;
pub mod gcs;
pub mod local_fs;
pub(crate) mod options;
mod retry;
pub mod s3;

pub use azure::AzureStorageProvider;
pub use gcs::GcsStorageProvider;
pub use local_fs::LocalFsStorageProvider;
pub(crate) use options::StorageOptions;
pub use s3::S3StorageProvider;

/// Object metadata returned by HEAD, GET, and list operations.
///
/// `size` is the content length in bytes. `etag` is the backend's
/// opaque version token (S3 ETag, LocalFS mtime-derived); used by
/// [`StorageProvider::put_if_match`] for CAS-fenced writes.
/// `last_modified` is `UNIX_EPOCH` for providers that don't surface it.
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub size: u64,
    pub etag: Option<String>,
    pub last_modified: SystemTime,
}

/// Errors surfaced by [`StorageProvider`] implementations.
///
/// Variants distinguish permanent failures from
/// transient-exhausted ones so callers can choose recovery
/// (typically none — retry is the provider's job).
#[derive(Debug, Error)]
pub enum StorageError {
    /// Object doesn't exist. Permanent. Returned by `head`,
    /// `get`, `get_range` against a missing URI. `delete` is
    /// idempotent — a missing target returns `Ok(())` rather
    /// than this variant.
    #[error("not found: {uri}")]
    NotFound { uri: String },

    /// Conditional write didn't satisfy precondition.
    ///
    /// Fires when `put_atomic` finds the target already exists
    /// (`If-None-Match: *` on S3, `O_EXCL` on LocalFS) or when
    /// `put_if_match` finds an ETag mismatch. The supertable's
    /// commit loop catches this on the pointer-CAS path and
    /// re-reads + retries; other callers surface it.
    #[error("precondition failed: {uri}")]
    PreconditionFailed { uri: String },

    /// Transient failure that the provider's internal retry
    /// loop already exhausted (e.g., persistent 5xx, repeated
    /// connection reset). Callers do **not** retry.
    #[error("transient error after retry: {uri} — {source}")]
    TransientExhausted {
        uri: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Permanent failure (auth, schema/region mismatch,
    /// corrupted response, malformed URI). Callers do **not**
    /// retry.
    #[error("permanent error: {uri} — {source}")]
    Permanent {
        uri: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Storage backend abstraction.
///
/// Implementations wrap `object_store` crate types (or fakes
/// for tests) and expose the subset of operations the
/// supertable's persistence + disk-cache layers consume.
///
/// All methods are async. Implementations are `Send + Sync`
/// so `Arc<dyn StorageProvider>` can be shared across the
/// supertable: the manifest part loader, the disk cache
/// store, and the writer all hold clones of the *same* `Arc`.
///
/// ## CAS-token invariant
///
/// A provider's conditional-write token is a single opaque,
/// backend-defined value. The token surfaced in [`ObjectMeta::etag`]
/// by `head`/`get`, the token returned by `put_atomic` /
/// `put_if_match`, and the token accepted by `put_if_match`'s
/// `expected_etag` are all the **same kind** (S3/Azure: the HTTP ETag;
/// GCS: the object generation). Callers chain the *returned* token
/// into the next `put_if_match` without re-reading, so a provider that
/// returns a different token kind than it accepts silently breaks OCC.
/// The `cas_conformance` test helper enforces this against every
/// backend.
#[async_trait]
pub trait StorageProvider: Send + Sync + fmt::Debug {
    /// Cheap metadata lookup. Used by the cold-fetch
    /// coordinator to size the destination file before
    /// issuing range-GETs.
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError>;

    /// Read the entire object together with its metadata. The
    /// returned [`ObjectMeta`] reflects the exact version whose
    /// bytes are in the response — no HEAD-then-GET race window
    /// — so callers chaining CAS writes against this read can
    /// use `meta.etag` directly.
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError>;

    /// Conditional read — like [`Self::get`], but returns `Ok(None)`
    /// ("not modified") when the object's current etag equals `etag`,
    /// and the full body + metadata otherwise.
    ///
    /// The default implementation issues a plain [`Self::get`] and
    /// compares etags locally — correct on every backend, though it
    /// still transfers the body. HTTP backends (S3, Azure) override
    /// it with a native `If-None-Match` request so the "unchanged"
    /// case is a bodyless 304. Callers use this for high-frequency
    /// freshness probes of tiny objects (the manifest pointer).
    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        let (bytes, meta) = self.get(uri).await?;
        if meta.etag.as_deref() == Some(etag) {
            return Ok(None);
        }
        Ok(Some((bytes, meta)))
    }

    /// Range-fetch. `range.end` is exclusive.
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError>;

    /// Tail-fetch path: — fetch the last `len` bytes of `uri` AND
    /// return the total object size from the same response.
    ///
    /// Lets cold-open callers (parquet footer / format trailer
    /// readers) skip an upfront `head()` round-trip: a single
    /// suffix-range GET pulls the bytes and discloses the
    /// object size at once.
    ///
    /// Implementations backed by HTTP range-GETs (S3, GCS)
    /// should use `Range: bytes=-len` so the response's
    /// Content-Range header carries the total size. The
    /// default impl falls back to a `head()` + bounded
    /// `get_range()` pair (one HEAD + one GET = 2 RTTs) for
    /// providers that can't directly issue a suffix range.
    ///
    /// `len` is clamped to the object size: callers requesting
    /// more bytes than the object holds receive the whole
    /// object plus `size == object_size`.
    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let meta = self.head(uri).await?;
        let len = len.min(meta.size);
        if len == 0 {
            return Ok((Bytes::new(), meta.size));
        }
        let start = meta.size - len;
        let bytes = self.get_range(uri, start..meta.size).await?;
        Ok((bytes, meta.size))
    }

    /// Atomic write — succeeds only if the target doesn't
    /// exist. Maps to `If-None-Match: *` on S3,
    /// `x-goog-if-generation-match: 0` on GCS, `O_EXCL` on
    /// LocalFS.
    ///
    /// Returns the new object's etag when the backend surfaces
    /// one (S3 always, LocalFs via mtime). `Ok(None)` is legal
    /// and means the write succeeded but no etag was reported;
    /// CAS-chained callers treat `None` as "create-only-if-
    /// absent" on the subsequent [`put_if_match`].
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError>;

    /// Conditional write — succeeds only if the target's
    /// current ETag matches `expected_etag`.
    ///
    /// Used for OCC on the manifest pointer: the supertable
    /// reads the current pointer (capturing its etag), builds
    /// the new manifest, then writes the new pointer with the
    /// old etag as precondition. A concurrent writer that
    /// commits between read and write causes
    /// `PreconditionFailed`, which the commit loop catches and
    /// retries.
    ///
    /// `None` expected etag means "create only if absent"
    /// (semantically identical to `put_atomic`); pass `Some`
    /// to update an existing object.
    ///
    /// Returns the new object's etag on success — same
    /// `Ok(None)` semantics as [`put_atomic`].
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError>;

    /// Streaming multipart upload — for superfiles larger than
    /// `SupertableOptions::put_multipart_threshold_bytes`
    /// (default 100 MB), the writer routes through this path
    /// instead of `put_atomic` to avoid buffering the whole
    /// superfile in RAM during commit.
    ///
    /// Returns the underlying `object_store::MultipartUpload`
    /// handle; callers drive it via its own `put_part` /
    /// `complete` / `abort` methods.
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError>;

    /// Delete an object. **Idempotent** — deleting a missing
    /// object returns `Ok(())`, not [`StorageError::NotFound`].
    async fn delete(&self, uri: &str) -> Result<(), StorageError>;

    /// List object URIs under `prefix`. Returns the full URI of
    /// every object whose path starts with `prefix` (caller is
    /// responsible for slash-aware boundary checks if they want
    /// to restrict to direct children).
    ///
    /// Used by the WAL recovery sweep to enumerate
    /// `wal/mutations/*.json`. Listing is a relatively heavy
    /// operation on object-store backends (it's a LIST call;
    /// pagination handled internally) so callers should not
    /// invoke this on the hot path — it's an open-time / sweep-
    /// time primitive.
    ///
    /// List objects under `prefix`, returning each key with its metadata.
    ///
    /// Default returns an empty list — test/mock providers that don't
    /// need listing can leave the default in place; production providers
    /// (LocalFs, S3, Azure, GCS) override.
    async fn list_with_prefix_metadata(
        &self,
        _prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        Ok(Vec::new())
    }

    /// List object keys under `prefix`. Derived from [`list_with_prefix_metadata`].
    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        Ok(self
            .list_with_prefix_metadata(prefix)
            .await?
            .into_iter()
            .map(|(key, _)| key)
            .collect())
    }

    /// Expose the underlying `object_store` handle plus the object
    /// key that `uri` maps to within it, when this provider is backed
    /// by a store DataFusion can range-GET directly.
    ///
    /// Used by the SQL scan and search-hit row resolution to hand
    /// DataFusion's `ParquetSource` the real object store so it issues
    /// async footer / row-group / page range GETs against object
    /// storage, instead of buffering whole superfiles into memory.
    ///
    /// `None` for providers without a native `object_store` handle
    /// (mocks / in-memory test doubles); those callers fall back to the
    /// whole-object read path.
    fn object_store_handle(
        &self,
        _uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, error::Error, ops::Range, sync::Mutex};

    use async_trait::async_trait;
    use bytes::Bytes;

    use super::*;

    /// Fixed etag the mock reports for any stored object.
    const MOCK_ETAG: &str = "mock-etag";

    /// Minimal in-memory [`StorageProvider`] implementing only the
    /// required methods, leaving `tail`, `list_with_prefix`, and
    /// `object_store_handle` at their trait defaults — those default
    /// bodies are the code under test here, since every production
    /// provider overrides all three.
    #[derive(Debug, Default)]
    struct InMemoryMock {
        objects: Mutex<HashMap<String, Bytes>>,
    }

    impl InMemoryMock {
        fn with(uri: &str, bytes: &[u8]) -> Self {
            let mock = Self::default();
            mock.objects
                .lock()
                .expect("lock")
                .insert(uri.into(), Bytes::copy_from_slice(bytes));
            mock
        }
    }

    fn not_found(uri: &str) -> StorageError {
        StorageError::NotFound { uri: uri.into() }
    }

    #[async_trait]
    impl StorageProvider for InMemoryMock {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok(ObjectMeta {
                    size: b.len() as u64,
                    etag: Some(MOCK_ETAG.into()),
                    last_modified: SystemTime::UNIX_EPOCH,
                }),
                None => Err(not_found(uri)),
            }
        }

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok((
                    b.clone(),
                    ObjectMeta {
                        size: b.len() as u64,
                        etag: Some(MOCK_ETAG.into()),
                        last_modified: SystemTime::UNIX_EPOCH,
                    },
                )),
                None => Err(not_found(uri)),
            }
        }

        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok(b.slice(range.start as usize..range.end as usize)),
                None => Err(not_found(uri)),
            }
        }

        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            let mut map = self.objects.lock().expect("lock");
            if map.contains_key(uri) {
                return Err(StorageError::PreconditionFailed { uri: uri.into() });
            }
            map.insert(uri.into(), bytes);
            Ok(Some(MOCK_ETAG.into()))
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            _expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            self.objects.lock().expect("lock").insert(uri.into(), bytes);
            Ok(Some(MOCK_ETAG.into()))
        }

        async fn put_multipart(
            &self,
            uri: &str,
        ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
            // The mock doesn't support streaming uploads; a permanent
            // error is enough to exercise the call path.
            let boxed: Box<dyn Error + Send + Sync> = "multipart unsupported".into();
            Err(StorageError::Permanent {
                uri: uri.into(),
                source: boxed,
            })
        }

        async fn delete(&self, uri: &str) -> Result<(), StorageError> {
            self.objects.lock().expect("lock").remove(uri);
            Ok(())
        }
    }

    // ---- default `tail` body (LocalFs aside, this is the fallback) ----

    #[tokio::test]
    async fn default_tail_returns_trailing_bytes_and_size() {
        let mock = InMemoryMock::with("k", b"abcdefgh");
        let (bytes, size) = mock.tail("k", 3).await.expect("tail");
        assert_eq!(size, 8);
        assert_eq!(&bytes[..], b"fgh");
    }

    #[tokio::test]
    async fn default_tail_clamps_len_to_object_size() {
        let mock = InMemoryMock::with("k", b"abc");
        let (bytes, size) = mock.tail("k", 100).await.expect("tail over-long");
        assert_eq!(size, 3);
        assert_eq!(&bytes[..], b"abc", "len clamps to the whole object");
    }

    #[tokio::test]
    async fn default_tail_zero_len_returns_empty_with_size() {
        let mock = InMemoryMock::with("k", b"abc");
        let (bytes, size) = mock.tail("k", 0).await.expect("tail zero");
        assert_eq!(size, 3);
        assert!(bytes.is_empty(), "zero-len tail still discloses size");
    }

    #[tokio::test]
    async fn default_tail_propagates_not_found() {
        let mock = InMemoryMock::default();
        assert!(matches!(
            mock.tail("missing", 4).await,
            Err(StorageError::NotFound { .. })
        ));
    }

    // ---- default `list_with_prefix` + `object_store_handle` ----

    #[tokio::test]
    async fn default_list_with_prefix_is_empty() {
        let mock = InMemoryMock::with("a/b", b"x");
        assert!(
            mock.list_with_prefix("a/").await.expect("list").is_empty(),
            "the default list never enumerates objects",
        );
    }

    #[test]
    fn default_object_store_handle_is_none() {
        let mock = InMemoryMock::default();
        assert!(mock.object_store_handle("k").is_none());
    }

    // ---- exercise the required methods so the mock's own surface is
    //      covered too (and the byte ops behave as the trait specifies) ----

    #[tokio::test]
    async fn mock_byte_ops_round_trip() {
        let mock = InMemoryMock::default();

        // put_atomic creates; a second create hits the precondition.
        assert_eq!(
            mock.put_atomic("k", Bytes::from_static(b"hello"))
                .await
                .expect("put_atomic"),
            Some(MOCK_ETAG.to_string()),
        );
        assert!(matches!(
            mock.put_atomic("k", Bytes::from_static(b"x")).await,
            Err(StorageError::PreconditionFailed { .. })
        ));

        // head + get + get_range read it back.
        assert_eq!(mock.head("k").await.expect("head").size, 5);
        let (bytes, _) = mock.get("k").await.expect("get");
        assert_eq!(&bytes[..], b"hello");
        assert_eq!(&mock.get_range("k", 1..3).await.expect("range")[..], b"el");

        // put_if_match overwrites unconditionally for the mock.
        mock.put_if_match("k", Bytes::from_static(b"world!"), Some(MOCK_ETAG))
            .await
            .expect("put_if_match");
        assert_eq!(mock.head("k").await.expect("head2").size, 6);

        // delete is idempotent.
        mock.delete("k").await.expect("delete");
        mock.delete("k").await.expect("delete idempotent");
        assert!(matches!(
            mock.get("k").await,
            Err(StorageError::NotFound { .. })
        ));
        assert!(matches!(
            mock.head("missing").await,
            Err(StorageError::NotFound { .. })
        ));
        assert!(matches!(
            mock.get_range("missing", 0..1).await,
            Err(StorageError::NotFound { .. })
        ));
    }

    #[tokio::test]
    async fn mock_put_multipart_surfaces_permanent_error() {
        let mock = InMemoryMock::default();
        assert!(matches!(
            mock.put_multipart("k").await,
            Err(StorageError::Permanent { .. })
        ));
    }

    // ---- error Display + ObjectMeta derives ----

    #[test]
    fn storage_error_display_covers_every_variant() {
        let cases: [(StorageError, &str); 4] = [
            (StorageError::NotFound { uri: "u".into() }, "not found"),
            (
                StorageError::PreconditionFailed { uri: "u".into() },
                "precondition failed",
            ),
            (
                StorageError::TransientExhausted {
                    uri: "u".into(),
                    source: "boom".into(),
                },
                "transient",
            ),
            (
                StorageError::Permanent {
                    uri: "u".into(),
                    source: "boom".into(),
                },
                "permanent",
            ),
        ];
        for (err, needle) in cases {
            assert!(
                err.to_string().contains(needle),
                "{err:?} display should contain {needle:?}",
            );
        }
    }

    #[test]
    fn object_meta_is_clone_and_debug() {
        let meta = ObjectMeta {
            size: 7,
            etag: Some("e".into()),
            last_modified: SystemTime::UNIX_EPOCH,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.size, 7);
        assert_eq!(cloned.etag.as_deref(), Some("e"));
        assert!(format!("{meta:?}").contains("ObjectMeta"));
    }
}
