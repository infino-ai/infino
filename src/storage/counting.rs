// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Meters for the raw [`ObjectStore`] handle and multipart uploads.
//!
//! Records into the same [`UsageMeter`] as [`crate::storage::StorageProvider`] methods
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

use crate::runtime_metrics::io::UsageMeter;

/// Wrap an [`ObjectStore`] so every successful read increments `meter`.
pub(crate) fn wrap_object_store(
    inner: Arc<dyn ObjectStore>,
    meter: Arc<UsageMeter>,
) -> Arc<dyn ObjectStore> {
    Arc::new(CountingObjectStore { inner, meter })
}

/// Wrap a multipart upload so each successful `put_part` / `complete`
/// records a PUT. Create is counted by the caller after
/// `put_multipart` / `put_multipart_opts` succeeds (`meter.record_put(0)`).
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
        let len = payload.content_length() as u64;
        let result = self.inner.put_opts(location, payload, opts).await?;
        self.meter.record_put(len);
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        let upload = self.inner.put_multipart_opts(location, opts).await?;
        // CreateMultipartUpload is billable; count only after the session exists.
        self.meter.record_put(0);
        Ok(wrap_multipart(upload, Arc::clone(&self.meter)))
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
        // A LIST request is issued when the stream is created; count it so
        // callers that go through `object_store_handle` share the ledger.
        self.meter.record_list();
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&ObjPath>) -> ObjectStoreResult<ListResult> {
        let result = self.inner.list_with_delimiter(prefix).await?;
        self.meter.record_list();
        Ok(result)
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
        let len = data.content_length() as u64;
        let meter = Arc::clone(&self.meter);
        let upload = self.inner.put_part(data);
        Box::pin(async move {
            let result = upload.await;
            if result.is_ok() {
                meter.record_put(len);
            }
            result
        })
    }

    async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
        let result = self.inner.complete().await;
        if result.is_ok() {
            self.meter.record_put(0);
        }
        result
    }

    async fn abort(&mut self) -> ObjectStoreResult<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, TryStreamExt, stream};
    use object_store::{ObjectStoreExt, memory::InMemory};

    use super::*;

    #[tokio::test]
    async fn object_store_wrapper_counts_lists_and_multipart_lifecycle() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let meter = UsageMeter::new();
        let counted = wrap_object_store(store, Arc::clone(&meter));

        // Debug/Display render without touching the inner store — these
        // strings end up in provider log lines.
        assert!(format!("{counted:?}").contains("CountingObjectStore"));
        assert!(format!("{counted}").contains("CountingObjectStore"));

        let a = ObjPath::from("seg/a.bin");
        counted
            .put(&a, PutPayload::from_static(b"abc"))
            .await
            .expect("put");

        // A streaming LIST bills once at stream creation;
        // list_with_delimiter bills a second request.
        let before = meter.snapshot();
        let listed: Vec<_> = counted.list(None).try_collect().await.expect("list");
        assert_eq!(listed.len(), 1);
        let flat = counted
            .list_with_delimiter(None)
            .await
            .expect("list_with_delimiter");
        // Delimiter semantics: `seg/a.bin` groups under the `seg`
        // common prefix rather than appearing as a top-level object.
        assert_eq!(flat.common_prefixes.len(), 1);
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.list_count, 2);

        // copy is an unbilled passthrough — no read or write we meter.
        let b = ObjPath::from("seg/b.bin");
        let before = meter.snapshot();
        counted.copy(&a, &b).await.expect("copy");
        assert!(meter.snapshot().since(&before).is_zero());

        // delete_stream passes through unbilled but must still delete.
        let victims = stream::iter(vec![Ok(b.clone())]).boxed();
        let deleted: Vec<_> = counted
            .delete_stream(victims)
            .try_collect()
            .await
            .expect("delete_stream");
        assert_eq!(deleted, vec![b.clone()]);
        assert!(counted.get(&b).await.is_err(), "b must be gone");

        // Multipart lifecycle: create bills a zero-byte PUT (the
        // CreateMultipartUpload request), each part bills its bytes, and
        // complete bills the finalize request.
        let m = ObjPath::from("seg/m.bin");
        let before = meter.snapshot();
        let mut upload = counted.put_multipart(&m).await.expect("create multipart");
        assert!(format!("{upload:?}").contains("CountingMultipart"));
        upload
            .put_part(PutPayload::from_static(b"0123456789"))
            .await
            .expect("part");
        upload.complete().await.expect("complete");
        let delta = meter.snapshot().since(&before);
        assert_eq!(
            delta.put_count, 3,
            "create + part + complete each bill one PUT"
        );
        assert_eq!(delta.put_bytes, 10, "only part bytes carry payload");

        // abort is an unbilled passthrough.
        let mut aborted = counted
            .put_multipart(&ObjPath::from("seg/n.bin"))
            .await
            .expect("create multipart");
        let before = meter.snapshot();
        aborted.abort().await.expect("abort");
        assert!(meter.snapshot().since(&before).is_zero());
    }

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

    #[tokio::test]
    async fn object_store_wrapper_counts_put_after_success() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjPath::from("seg/w.bin");
        let meter = UsageMeter::new();
        let counted = wrap_object_store(store, Arc::clone(&meter));
        let before = meter.snapshot();
        counted
            .put(&path, PutPayload::from_static(b"abcd"))
            .await
            .expect("put");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.put_count, 1);
        assert_eq!(delta.put_bytes, 4);
    }

    #[tokio::test]
    async fn multipart_part_failure_does_not_record_put() {
        fn not_implemented(operation: &str) -> object_store::Error {
            object_store::Error::NotImplemented {
                operation: operation.to_string(),
                implementer: "FailPartUpload".to_string(),
            }
        }

        /// Upload that fails every `put_part`.
        #[derive(Debug)]
        struct FailPartUpload;

        #[async_trait]
        impl MultipartUpload for FailPartUpload {
            fn put_part(&mut self, _data: PutPayload) -> UploadPart {
                Box::pin(async { Err(not_implemented("put_part")) })
            }
            async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
                Err(not_implemented("complete"))
            }
            async fn abort(&mut self) -> ObjectStoreResult<()> {
                Ok(())
            }
        }

        let meter = UsageMeter::new();
        let mut upload = wrap_multipart(Box::new(FailPartUpload), Arc::clone(&meter));
        let before = meter.snapshot();
        let err = upload
            .put_part(PutPayload::from_static(b"x"))
            .await
            .expect_err("part must fail");
        assert!(matches!(err, object_store::Error::NotImplemented { .. }));
        assert!(
            meter.snapshot().since(&before).is_zero(),
            "failed put_part must not bump the ledger"
        );
    }
}
