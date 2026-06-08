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

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use thiserror::Error;

pub mod azure;
pub mod local_fs;
mod retry;
pub mod s3;

pub use azure::AzureStorageProvider;
pub use local_fs::LocalFsStorageProvider;
pub use s3::S3StorageProvider;

/// Cheap object metadata — what HEAD returns.
///
/// `size` is the object's content length in bytes. `etag` is
/// the backend's opaque version identifier (S3 ETag, GCS
/// generation as a string, LocalFS mtime-derived token); used
/// by [`StorageProvider::put_if_match`] for CAS-fenced writes.
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub size: u64,
    pub etag: Option<String>,
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
#[async_trait]
pub trait StorageProvider: Send + Sync + std::fmt::Debug {
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

    /// Streaming multipart upload — for segments larger than
    /// `SupertableOptions::put_multipart_threshold_bytes`
    /// (default 100 MB), the writer routes through this path
    /// instead of `put_atomic` to avoid buffering the whole
    /// segment in RAM during commit.
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
    /// Default returns an empty list — test/mock providers that
    /// don't need WAL recovery support can leave the default in
    /// place; production providers (LocalFs, S3) override.
    async fn list_with_prefix(&self, _prefix: &str) -> Result<Vec<String>, StorageError> {
        Ok(Vec::new())
    }
}
