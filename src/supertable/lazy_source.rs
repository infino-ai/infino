// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable-side [`LazyByteSource`] implementations.
//!
//! The superfile crate owns the trait
//! ([`crate::superfile::LazyByteSource`]). The supertable
//! crate owns the impls that bridge to the storage layer:
//!
//! - [`StorageRangeSource`] wraps an
//!   `Arc<dyn StorageProvider>` so per-query callers can run
//!   `SuperfileReader::open_lazy` against any storage
//!   backend. This is the `ColdFetchMode::RangeOnly` path —
//!   stateless callers that don't want to materialize the
//!   segment in the disk cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};

use crate::storage::{StorageError, StorageProvider};
use crate::superfile::{LazyByteSource, LazyByteSourceError};

/// Bounded re-issue budget for a `get_range` that comes back
/// short of the requested length. Each re-issue fetches only the
/// still-missing tail, so a healthy backend completes on the
/// first retry; the cap stops a persistently-truncated object
/// from spinning forever before it surfaces a typed
/// [`LazyByteSourceError::ShortRead`].
const MAX_SHORT_READ_RETRIES: u32 = 4;

/// `LazyByteSource` over a `StorageProvider::get_range`.
///
/// Each call to [`range`] issues a fresh `get_range` against
/// the storage backend. Use this for stateless / RangeOnly
/// callers; for steady-state hot reads the disk-cache store
/// is the right path.
///
/// ## Size discovery
///
/// `size` is an `AtomicU64` rather than a plain
/// `u64` so the source can be constructed *without* an
/// up-front HEAD round-trip. The first call to [`tail`] (used
/// by cold-open callers like `read_parquet_metadata_lazy`)
/// issues a suffix-range GET, learns the size from the
/// response, and patches the atomic. Subsequent calls see
/// the cached value via `size()`.
///
/// When the size *is* known up-front (the disk-cache layer
/// already HEAD'd, or a sync test passes a known length),
/// [`Self::with_known_size`] populates the atomic at
/// construction so `range()` can still bounds-check.
///
/// [`range`]: LazyByteSource::range
/// [`tail`]: LazyByteSource::tail
#[derive(Debug)]
pub struct StorageRangeSource {
    storage: Arc<dyn StorageProvider>,
    /// Storage-side URI of the object (e.g.
    /// `data/seg-<uuid>.sf.parquet`).
    uri: String,
    /// Cached total size. `0` means "not yet known". Set
    /// either at construction ([`Self::with_known_size`] /
    /// [`Self::new`]) or lazily on the first [`tail`] call.
    ///
    /// [`tail`]: LazyByteSource::tail
    size: AtomicU64,
}

impl StorageRangeSource {
    /// Construct + cache the object's total size. One HEAD
    /// round-trip up-front; subsequent `range` calls each do
    /// their own GET-range.
    pub async fn new(
        storage: Arc<dyn StorageProvider>,
        uri: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let uri: String = uri.into();
        let meta = storage.head(&uri).await?;
        Ok(Self {
            storage,
            uri,
            size: AtomicU64::new(meta.size),
        })
    }

    /// Construct with a caller-provided size. Used by
    /// `DiskCacheStore::cold_fetch_lazy` when the cache layer
    /// has already issued a HEAD (callers that haven't
    /// prefer [`Self::with_unknown_size`] to skip the HEAD
    /// entirely).
    pub fn with_known_size(
        storage: Arc<dyn StorageProvider>,
        uri: impl Into<String>,
        size: u64,
    ) -> Self {
        Self {
            storage,
            uri: uri.into(),
            size: AtomicU64::new(size),
        }
    }

    /// construct without an up-front size.
    ///
    /// The size is discovered lazily on the first
    /// [`LazyByteSource::tail`] call (which uses a native
    /// suffix-range GET that returns size in the response).
    /// Callers that rely on `size()` being non-zero before
    /// any I/O happens must use [`Self::new`] or
    /// [`Self::with_known_size`] instead.
    ///
    /// Cold-open is the canonical caller: it starts with a
    /// parquet-footer `tail()` call which both fetches the
    /// bytes and patches the size in one round-trip,
    /// saving an entire HEAD vs. [`Self::new`].
    pub fn with_unknown_size(storage: Arc<dyn StorageProvider>, uri: impl Into<String>) -> Self {
        Self {
            storage,
            uri: uri.into(),
            size: AtomicU64::new(0),
        }
    }

    /// Storage URI this source pulls from. Useful for tests
    /// and observability.
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

#[async_trait]
impl LazyByteSource for StorageRangeSource {
    fn size(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let known = self.size.load(Ordering::Acquire);
        // Only bounds-check when the size is known. With
        // `with_unknown_size`, the first range call may
        // legitimately precede the discovery `tail()`; we
        // trust the underlying storage to surface OOB as a
        // typed `StorageError`.
        if known > 0 && start.saturating_add(len) > known {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: known,
            });
        }
        if len == 0 {
            return Ok(Bytes::new());
        }

        // Completion loop. `StorageProvider::get_range` is
        // contractually exact-length-or-error, but object-store
        // backends have been observed to return a *short* buffer
        // without erroring — a clamped/partial range on a transient
        // transport hiccup, or an object shorter than the cached
        // size. The `LazyByteSource::range` contract requires the
        // returned bytes to equal `full_object[start..start + len]`,
        // so re-issue the GET for the still-missing tail rather than
        // handing a truncated buffer up to the sub-readers (where a
        // short slice panics deep in the vector/FTS codec). A read
        // that makes no forward progress, or stalls past the retry
        // budget, surfaces as a typed `ShortRead`.
        let want = len as usize;
        let end = start + len;
        let mut cursor = start;
        let mut filled = 0usize;
        let mut parts: Vec<Bytes> = Vec::new();
        let mut stalls = 0u32;
        while filled < want {
            // `StorageError` -> `LazyByteSourceError::Storage`
            // via the `#[from]` impl — typed propagation, no
            // stringification.
            let chunk = self.storage.get_range(&self.uri, cursor..end).await?;
            if chunk.is_empty() {
                // No bytes and no error means the object is shorter
                // than the requested range; looping again would spin.
                return Err(LazyByteSourceError::ShortRead {
                    start,
                    requested: len,
                    got: filled as u64,
                });
            }
            // Defensive clamp: a backend must never overshoot the
            // requested tail, but never trust more than `want`.
            let take = chunk.len().min(want - filled);
            filled += take;
            cursor += take as u64;
            parts.push(chunk.slice(0..take));
            if filled < want {
                stalls += 1;
                if stalls > MAX_SHORT_READ_RETRIES {
                    return Err(LazyByteSourceError::ShortRead {
                        start,
                        requested: len,
                        got: filled as u64,
                    });
                }
            }
        }

        // Fast path: a single full-length chunk (the overwhelming
        // common case) returns zero-copy.
        if parts.len() == 1 {
            return Ok(parts.pop().expect("len checked == 1"));
        }
        let mut out = BytesMut::with_capacity(want);
        for p in &parts {
            out.extend_from_slice(p);
        }
        Ok(out.freeze())
    }

    /// single-RTT tail fetch.
    ///
    /// Routes through `StorageProvider::tail`, which on S3
    /// uses a native suffix-range GET so the response carries
    /// both the bytes AND the total object size. The first
    /// `tail()` call on a [`Self::with_unknown_size`] source
    /// patches the cached size atomic, so subsequent
    /// `range()` callers get the same bounds-checking
    /// behavior as if the source had been constructed with
    /// a known size.
    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        let (bytes, total) = self.storage.tail(&self.uri, len).await?;
        // Patch the size atomic if this was the first call
        // against an `with_unknown_size` source. Use
        // `store(Release)` rather than CAS — concurrent
        // `tail` calls would all observe the same total, so
        // a last-writer-wins store is correct.
        self.size.store(total, Ordering::Release);
        Ok((bytes, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ObjectMeta;
    use std::sync::atomic::AtomicUsize;

    /// Storage fake that serves `get_range` in capped chunks and
    /// against a (possibly smaller-than-advertised) backing object.
    /// Models object-store backends that return a short buffer for an
    /// in-bounds request without erroring.
    #[derive(Debug)]
    struct ChunkedStorage {
        blob: Bytes,
        /// Largest number of bytes a single `get_range` returns. A
        /// value `< requested len` forces the completion loop to
        /// re-issue for the missing tail.
        chunk_cap: usize,
        /// Bytes actually present in the backing object. Requests past
        /// this clamp to it (mimicking S3 clamping to object size).
        obj_size: usize,
        calls: AtomicUsize,
    }

    impl ChunkedStorage {
        fn new(blob: Bytes, chunk_cap: usize, obj_size: usize) -> Self {
            Self {
                blob,
                chunk_cap,
                obj_size,
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    fn permanent(uri: &str, msg: &'static str) -> StorageError {
        let boxed: Box<dyn std::error::Error + Send + Sync> = msg.into();
        StorageError::Permanent {
            uri: uri.into(),
            source: boxed,
        }
    }

    #[async_trait]
    impl StorageProvider for ChunkedStorage {
        async fn head(&self, _uri: &str) -> Result<ObjectMeta, StorageError> {
            Ok(ObjectMeta {
                size: self.obj_size as u64,
                etag: None,
            })
        }

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            Err(permanent(uri, "get unimplemented"))
        }

        async fn get_range(
            &self,
            _uri: &str,
            range: std::ops::Range<u64>,
        ) -> Result<Bytes, StorageError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            let start = range.start as usize;
            let req = (range.end - range.start) as usize;
            let available = self.obj_size.saturating_sub(start);
            let take = req.min(self.chunk_cap).min(available);
            Ok(self.blob.slice(start..start + take))
        }

        async fn put_atomic(
            &self,
            uri: &str,
            _bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            Err(permanent(uri, "put_atomic unimplemented"))
        }

        async fn put_if_match(
            &self,
            uri: &str,
            _bytes: Bytes,
            _expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            Err(permanent(uri, "put_if_match unimplemented"))
        }

        async fn put_multipart(
            &self,
            uri: &str,
        ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
            Err(permanent(uri, "put_multipart unimplemented"))
        }

        async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
            Ok(())
        }
    }

    /// A `get_range` that comes back short (capped chunks) must be
    /// completed by re-issuing for the missing tail — the caller sees
    /// the full, contiguous range, never a truncated buffer.
    #[tokio::test]
    async fn range_completes_a_short_read_by_refetching_the_tail() {
        let blob = Bytes::from((0u8..=255).cycle().take(4096).collect::<Vec<u8>>());
        // Each GET returns at most 1000 bytes; the full object is
        // present. A 4096-byte request must stitch ≥5 chunks.
        let storage = Arc::new(ChunkedStorage::new(blob.clone(), 1000, blob.len()));
        let src = StorageRangeSource::with_known_size(
            Arc::clone(&storage) as Arc<dyn StorageProvider>,
            "seg.sf.parquet",
            blob.len() as u64,
        );

        let got = src.range(0, blob.len() as u64).await.expect("range");
        assert_eq!(got.len(), blob.len());
        assert_eq!(got.as_ref(), blob.as_ref());
        assert!(
            storage.call_count() >= 5,
            "expected multiple GETs to complete the short read, got {}",
            storage.call_count()
        );
    }

    /// Stitching works for an interior, non-zero-based range too.
    #[tokio::test]
    async fn range_completes_short_read_for_interior_range() {
        let blob = Bytes::from((0u8..=255).cycle().take(4096).collect::<Vec<u8>>());
        let storage = Arc::new(ChunkedStorage::new(blob.clone(), 700, blob.len()));
        let src = StorageRangeSource::with_known_size(
            Arc::clone(&storage) as Arc<dyn StorageProvider>,
            "seg.sf.parquet",
            blob.len() as u64,
        );
        let (start, len) = (1024u64, 2048u64);
        let got = src.range(start, len).await.expect("range");
        assert_eq!(got.as_ref(), &blob[start as usize..(start + len) as usize]);
    }

    /// When the backing object is genuinely shorter than the cached
    /// size (a stale/oversized size hint), the read can never be
    /// completed — it must surface a typed `ShortRead`, not a
    /// truncated buffer that panics downstream.
    #[tokio::test]
    async fn range_surfaces_short_read_when_object_is_truncated() {
        let blob = Bytes::from(vec![7u8; 2048]);
        // Cached size says 4096, but the object only holds 2048.
        let storage = Arc::new(ChunkedStorage::new(blob, 4096, 2048));
        let src = StorageRangeSource::with_known_size(
            Arc::clone(&storage) as Arc<dyn StorageProvider>,
            "seg.sf.parquet",
            4096,
        );
        let err = src
            .range(0, 4096)
            .await
            .expect_err("must reject a permanently short read");
        match err {
            LazyByteSourceError::ShortRead {
                start,
                requested,
                got,
            } => {
                assert_eq!(start, 0);
                assert_eq!(requested, 4096);
                assert_eq!(got, 2048);
            }
            other => panic!("expected ShortRead, got {other:?}"),
        }
    }

    /// A zero-length range is a no-op that never touches storage.
    #[tokio::test]
    async fn range_zero_length_is_empty_without_io() {
        let storage = Arc::new(ChunkedStorage::new(Bytes::from(vec![0u8; 16]), 16, 16));
        let src = StorageRangeSource::with_known_size(
            Arc::clone(&storage) as Arc<dyn StorageProvider>,
            "seg.sf.parquet",
            16,
        );
        let got = src.range(8, 0).await.expect("zero-length range");
        assert!(got.is_empty());
        assert_eq!(storage.call_count(), 0);
    }
}
