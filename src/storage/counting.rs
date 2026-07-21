// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-side meters for the raw [`ObjectStore`] handle and multipart
//! uploads.
//!
//! [`StorageProvider`] methods record into [`super::io_counters`] directly.
//! Parquet / DataFusion range GETs go through [`StorageProvider::object_store_handle`]
//! and would otherwise bypass those hooks — this wrapper closes that hole.
//! Multipart part + complete calls similarly never hit `put_atomic`, so they
//! are wrapped here too.
//!
//! Overhead is a handful of `AtomicU64` relaxed adds per successful op —
//! negligible next to object-store RTT.

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta as OsObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as ObjectStoreResult, UploadPart, path::Path as ObjPath,
};

use super::io_counters;

/// Wrap an [`ObjectStore`] so every successful read increments engine
/// [`io_counters`]. Base providers use this for
/// [`super::StorageProvider::object_store_handle`].
pub(crate) fn wrap_object_store(inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
    Arc::new(CountingObjectStore { inner })
}

/// Layer a hidden-namespace tag on top of an already-counting store.
///
/// [`super::PrefixedStorageProvider`] uses this so parquet range GETs under
/// the hidden vector index increment [`io_counters::record_hidden_get`]
/// without double-counting the total GET (the inner wrap already called
/// [`io_counters::record_get`]).
pub(crate) fn tag_hidden_object_store(inner: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
    Arc::new(HiddenTaggingObjectStore { inner })
}

/// Wrap a multipart upload so each `put_part` / `complete` records a PUT
/// (matching S3 UploadPart + CompleteMultipartUpload billing). The Create
/// Multipart Upload request is counted by the caller via
/// [`io_counters::record_put`]`(0)`. Aborts are left uncounted.
pub(crate) fn wrap_multipart(inner: Box<dyn MultipartUpload>) -> Box<dyn MultipartUpload> {
    Box::new(CountingMultipart { inner })
}

struct CountingObjectStore {
    inner: Arc<dyn ObjectStore>,
}

impl fmt::Debug for CountingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingObjectStore")
            .finish_non_exhaustive()
    }
}

impl fmt::Display for CountingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingObjectStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for CountingObjectStore {
    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> ObjectStoreResult<GetResult> {
        let is_head = options.head;
        let res = self.inner.get_opts(location, options).await?;
        if is_head {
            io_counters::record_head();
        } else {
            let len = res.range.end.saturating_sub(res.range.start);
            io_counters::record_get(len);
        }
        Ok(res)
    }

    async fn get_ranges(
        &self,
        location: &ObjPath,
        ranges: &[std::ops::Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        let out = self.inner.get_ranges(location, ranges).await?;
        for b in &out {
            io_counters::record_get(b.len() as u64);
        }
        Ok(out)
    }

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, ObjectStoreResult<ObjPath>>,
    ) -> BoxStream<'static, ObjectStoreResult<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, ObjectStoreResult<OsObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&ObjPath>) -> ObjectStoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjPath,
        to: &ObjPath,
        options: CopyOptions,
    ) -> ObjectStoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// Counts only the hidden tag; total GETs come from an inner
/// [`CountingObjectStore`] (or equivalent provider-level `record_get`).
struct HiddenTaggingObjectStore {
    inner: Arc<dyn ObjectStore>,
}

impl fmt::Debug for HiddenTaggingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HiddenTaggingObjectStore")
            .finish_non_exhaustive()
    }
}

impl fmt::Display for HiddenTaggingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HiddenTaggingObjectStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for HiddenTaggingObjectStore {
    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> ObjectStoreResult<GetResult> {
        let is_head = options.head;
        let res = self.inner.get_opts(location, options).await?;
        if !is_head {
            let len = res.range.end.saturating_sub(res.range.start);
            io_counters::record_hidden_get(len);
        }
        Ok(res)
    }

    async fn get_ranges(
        &self,
        location: &ObjPath,
        ranges: &[std::ops::Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        let out = self.inner.get_ranges(location, ranges).await?;
        for b in &out {
            io_counters::record_hidden_get(b.len() as u64);
        }
        Ok(out)
    }

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, ObjectStoreResult<ObjPath>>,
    ) -> BoxStream<'static, ObjectStoreResult<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, ObjectStoreResult<OsObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&ObjPath>) -> ObjectStoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjPath,
        to: &ObjPath,
        options: CopyOptions,
    ) -> ObjectStoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

struct CountingMultipart {
    inner: Box<dyn MultipartUpload>,
}

impl fmt::Debug for CountingMultipart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingMultipart").finish_non_exhaustive()
    }
}

#[async_trait]
impl MultipartUpload for CountingMultipart {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        io_counters::record_put(data.content_length() as u64);
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
        io_counters::record_put(0);
        self.inner.complete().await
    }

    async fn abort(&mut self) -> ObjectStoreResult<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;

    #[tokio::test]
    async fn object_store_wrapper_counts_get_and_hidden_tag_layers() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjPath::from("seg/x.bin");
        store
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .expect("put");

        let before = io_counters::snapshot();
        let counted = wrap_object_store(Arc::clone(&store));
        let bytes = counted
            .get(&path)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("body");
        assert_eq!(bytes.as_ref(), b"0123456789");
        let delta = io_counters::snapshot().since(&before);
        // Process-global counters — use lower bounds under parallel tests.
        assert!(delta.get_count >= 1);
        assert!(delta.get_bytes >= 10);

        let before = io_counters::snapshot();
        let hidden = tag_hidden_object_store(counted);
        let _ = hidden.get(&path).await.expect("get");
        let delta = io_counters::snapshot().since(&before);
        // Inner counting wrap + outer hidden tag: total GET and hidden both advance.
        assert!(delta.get_count >= 1);
        assert!(delta.hidden_get_count >= 1);
        assert!(delta.hidden_get_bytes >= 10);
    }
}
