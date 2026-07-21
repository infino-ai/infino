// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Meters for the raw [`ObjectStore`] handle and multipart uploads.
//!
//! Records into the same [`UsageMeter`] as [`super::StorageProvider`] methods
//! so parquet range GETs and multipart parts share one ledger.

use std::{fmt, ops::Range, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta as OsObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as ObjectStoreResult, UploadPart, path::Path as ObjPath,
};

use super::usage::UsageMeter;

/// Wrap an [`ObjectStore`] so every successful read increments `meter`.
pub(crate) fn wrap_object_store(
    inner: Arc<dyn ObjectStore>,
    meter: Arc<UsageMeter>,
) -> Arc<dyn ObjectStore> {
    Arc::new(CountingObjectStore { inner, meter })
}

/// Wrap a multipart upload so each `put_part` / `complete` records a PUT.
/// Create is counted by the caller via `meter.record_put(0)`.
pub(crate) fn wrap_multipart(
    inner: Box<dyn MultipartUpload>,
    meter: Arc<UsageMeter>,
) -> Box<dyn MultipartUpload> {
    Box::new(CountingMultipart { inner, meter })
}

struct CountingObjectStore {
    inner: Arc<dyn ObjectStore>,
    meter: Arc<UsageMeter>,
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
            self.meter.record_head();
        } else {
            let len = res.range.end.saturating_sub(res.range.start);
            self.meter.record_get(
                location.as_ref(),
                Some((res.range.start, res.range.end)),
                len,
            );
        }
        Ok(res)
    }

    async fn get_ranges(
        &self,
        location: &ObjPath,
        ranges: &[Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        let out = self.inner.get_ranges(location, ranges).await?;
        for (r, b) in ranges.iter().zip(&out) {
            self.meter
                .record_get(location.as_ref(), Some((r.start, r.end)), b.len() as u64);
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
    meter: Arc<UsageMeter>,
}

impl fmt::Debug for CountingMultipart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingMultipart").finish_non_exhaustive()
    }
}

#[async_trait]
impl MultipartUpload for CountingMultipart {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.meter.record_put(data.content_length() as u64);
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
        self.meter.record_put(0);
        self.inner.complete().await
    }

    async fn abort(&mut self) -> ObjectStoreResult<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn object_store_wrapper_counts_get() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjPath::from("seg/x.bin");
        store
            .put(&path, PutPayload::from_static(b"0123456789"))
            .await
            .expect("put");

        let meter = UsageMeter::new();
        let counted = wrap_object_store(store, Arc::clone(&meter));
        let before = meter.snapshot();
        let bytes = counted
            .get(&path)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("body");
        assert_eq!(bytes.as_ref(), b"0123456789");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.get_bytes, 10);
    }
}
