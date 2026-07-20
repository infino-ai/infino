// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`SuperfileObjectStore`] — the thin DataFusion integration layer.
//!
//! SQL is the one query path that hands execution to DataFusion, and
//! DataFusion's `ParquetSource` reads its input through an
//! [`object_store::ObjectStore`]. This is that store — and *only* that.
//! It owns no storage policy: it serves byte ranges straight out of the
//! [`LazyByteSource`] each superfile's [`SuperfileReader::byte_source`]
//! already exposes. The provider calls `superfile_reader(...)`, takes
//! the byte source, registers it here, and DataFusion reads.
//!
//! There is exactly one read path and no branch on storage mode:
//!
//! - warm / mmap-backed superfiles resolve every range as a zero-copy
//!   `Bytes::slice` (the resident-bytes [`LazyByteSource`]); nothing is
//!   copied into a DataFusion `InMemory` store, so warm SQL is as cheap
//!   as the FTS/vector resolve path (slice = refcount bump).
//! - cold / lazy superfiles range-fetch straight from object storage
//!   through the same source.
//!
//! Only the read methods are real; this store is never written to,
//! listed, or copied during a scan, so the mutating trait methods
//! return [`object_store::Error::NotImplemented`].
//!
//! [`SuperfileReader::byte_source`]: crate::superfile::SuperfileReader::byte_source

#[cfg(test)]
use std::collections::HashMap;
use std::{fmt, ops::Range, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::stream::{self, BoxStream, StreamExt};
use object_store::{
    Attributes, CopyOptions, Error as OsError, GetOptions, GetRange, GetResult, GetResultPayload,
    ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, Result as OsResult, path::Path as ObjPath,
};

use crate::superfile::{LazyByteSource, lazy_source::Source};

/// Fixed `last_modified` reported for every registered superfile.
/// Superfiles are immutable once committed, so a wall-clock timestamp
/// carries no signal here — and a value that changed on every call
/// would defeat any downstream cache keyed on `(path, last_modified)`
/// and make responses non-deterministic.
const SUPERFILE_LAST_MODIFIED: DateTime<Utc> = DateTime::UNIX_EPOCH;

/// Read-only [`ObjectStore`] backed by per-superfile [`LazyByteSource`]s.
///
/// A pinned provider constructs one registry with [`Self::new`], fills it via
/// [`Self::insert_source`] as immutable files are first opened, and reuses it
/// for every DataFusion scan of that manifest.
pub(crate) struct SuperfileObjectStore {
    /// One byte source per surviving superfile, keyed by the same path
    /// used to build the superfile's `PartitionedFile`.
    sources: Arc<DashMap<ObjPath, Arc<dyn LazyByteSource>>>,
}

impl fmt::Debug for SuperfileObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuperfileObjectStore")
            .field("n_superfiles", &self.sources.len())
            .finish()
    }
}

impl fmt::Display for SuperfileObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SuperfileObjectStore({} superfiles)", self.sources.len())
    }
}

impl SuperfileObjectStore {
    /// Empty registry filled lazily as a pinned provider prepares files.
    pub(crate) fn new() -> Self {
        Self {
            sources: Arc::new(DashMap::new()),
        }
    }

    /// Build the store from the superfile byte sources gathered during a
    /// scan. Each key is the path the matching `PartitionedFile` is
    /// created with.
    #[cfg(test)]
    pub(crate) fn from_sources(sources: HashMap<ObjPath, Arc<dyn LazyByteSource>>) -> Self {
        let store = Self::new();
        for (path, source) in sources {
            store.insert_source(path, source);
        }
        store
    }

    /// Register or refresh one immutable superfile's byte source.
    pub(crate) fn insert_source(&self, location: ObjPath, source: Arc<dyn LazyByteSource>) {
        self.sources.insert(location, source);
    }

    fn source(&self, location: &ObjPath) -> OsResult<Arc<dyn LazyByteSource>> {
        self.sources
            .get(location)
            .map(|source| Arc::clone(source.value()))
            .ok_or_else(|| OsError::NotFound {
                path: location.to_string(),
                source: format!("superfile {location} not registered in SuperfileObjectStore")
                    .into(),
            })
    }
}

/// Resolve a [`GetRange`] (or its absence) into a concrete, clamped
/// `[start, end)` over an object of `size` bytes.
fn resolve_range(range: Option<GetRange>, size: u64) -> Range<u64> {
    match range {
        None => 0..size,
        Some(GetRange::Bounded(r)) => r.start.min(size)..r.end.min(size),
        Some(GetRange::Offset(start)) => start.min(size)..size,
        Some(GetRange::Suffix(n)) => size.saturating_sub(n)..size,
    }
}

fn not_implemented(operation: &str) -> OsError {
    OsError::NotImplemented {
        operation: operation.to_string(),
        implementer: "SuperfileObjectStore".to_string(),
    }
}

#[async_trait]
impl ObjectStore for SuperfileObjectStore {
    async fn get_opts(&self, location: &ObjPath, options: GetOptions) -> OsResult<GetResult> {
        let source = self.source(location)?;
        let size = source.size();
        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: SUPERFILE_LAST_MODIFIED,
            size,
            e_tag: None,
            version: None,
        };

        // A HEAD-style request only needs the metadata.
        if options.head {
            return Ok(GetResult {
                payload: GetResultPayload::Stream(stream::empty().boxed()),
                meta,
                range: 0..0,
                attributes: Attributes::default(),
            });
        }

        let range = resolve_range(options.range, size);
        let len = range.end.saturating_sub(range.start);
        let bytes = if len == 0 {
            Bytes::new()
        } else {
            // Route through `range_async` so a warm, mmap/block-resident scan
            // serves its bytes synchronously (zero-copy, no I/O) instead of
            // awaiting a range fetch; a cold miss still `await`s the GET.
            Source::Lazy(source)
                .range_async(range.start as usize..range.end as usize)
                .await
                .map_err(|e| OsError::Generic {
                    store: "SuperfileObjectStore",
                    source: Box::new(e),
                })?
        };

        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes: Attributes::default(),
        })
    }

    async fn put_opts(
        &self,
        _location: &ObjPath,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> OsResult<PutResult> {
        Err(not_implemented("put_opts"))
    }

    async fn put_multipart_opts(
        &self,
        _location: &ObjPath,
        _opts: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        Err(not_implemented("put_multipart_opts"))
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OsResult<ObjPath>>,
    ) -> BoxStream<'static, OsResult<ObjPath>> {
        locations
            .map(|_| Err(not_implemented("delete_stream")))
            .boxed()
    }

    fn list(&self, _prefix: Option<&ObjPath>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        stream::empty().boxed()
    }

    async fn list_with_delimiter(&self, _prefix: Option<&ObjPath>) -> OsResult<ListResult> {
        Ok(ListResult {
            common_prefixes: Vec::new(),
            objects: Vec::new(),
        })
    }

    async fn copy_opts(
        &self,
        _from: &ObjPath,
        _to: &ObjPath,
        _options: CopyOptions,
    ) -> OsResult<()> {
        Err(not_implemented("copy_opts"))
    }
}

#[cfg(test)]
mod tests {
    use object_store::ObjectStoreExt;

    use super::*;
    use crate::superfile::BytesLazyByteSource;

    fn store_with(path: &str, body: &'static [u8]) -> (SuperfileObjectStore, ObjPath) {
        let p = ObjPath::from(path);
        let mut sources: HashMap<ObjPath, Arc<dyn LazyByteSource>> = HashMap::new();
        sources.insert(
            p.clone(),
            Arc::new(BytesLazyByteSource::new(Bytes::from_static(body))),
        );
        (SuperfileObjectStore::from_sources(sources), p)
    }

    #[tokio::test]
    async fn serves_full_and_ranged_reads_zero_copy() {
        let (store, p) = store_with("seg-a.parquet", b"0123456789");

        let full = store
            .get(&p)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("bytes");
        assert_eq!(&full[..], b"0123456789");

        let mid = store.get_range(&p, 2..5).await.expect("range");
        assert_eq!(&mid[..], b"234");

        let head = store.head(&p).await.expect("head");
        assert_eq!(head.size, 10);
    }

    #[tokio::test]
    async fn offset_and_suffix_ranges_resolve_and_clamp() {
        let (store, p) = store_with("seg-a.parquet", b"0123456789");

        // Offset: from `start` to the end; a past-the-end start clamps
        // to an empty read instead of erroring.
        let tail = get_with_range(&store, &p, GetRange::Offset(7)).await;
        assert_eq!(&tail[..], b"789");
        let empty = get_with_range(&store, &p, GetRange::Offset(99)).await;
        assert!(empty.is_empty());

        // Suffix: the last `n` bytes; an oversized suffix clamps to the
        // whole object.
        let suffix = get_with_range(&store, &p, GetRange::Suffix(4)).await;
        assert_eq!(&suffix[..], b"6789");
        let all = get_with_range(&store, &p, GetRange::Suffix(99)).await;
        assert_eq!(&all[..], b"0123456789");
    }

    /// `get_opts` with an explicit [`GetRange`], collected to bytes.
    async fn get_with_range(store: &SuperfileObjectStore, p: &ObjPath, range: GetRange) -> Bytes {
        let options = GetOptions {
            range: Some(range),
            ..Default::default()
        };
        store
            .get_opts(p, options)
            .await
            .expect("get_opts")
            .bytes()
            .await
            .expect("bytes")
    }

    #[tokio::test]
    async fn unknown_path_is_not_found() {
        let (store, _p) = store_with("seg-a.parquet", b"abc");
        let err = store
            .get(&ObjPath::from("missing.parquet"))
            .await
            .expect_err("get of an unregistered path must fail");
        assert!(matches!(err, OsError::NotFound { .. }), "{err}");
    }

    #[tokio::test]
    async fn mutations_are_not_implemented() {
        let (store, p) = store_with("seg-a.parquet", b"abc");
        let err = store
            .put(&p, PutPayload::from_static(b"x"))
            .await
            .expect_err("writes to the read-only store must fail");
        assert!(matches!(err, OsError::NotImplemented { .. }), "{err}");
    }

    #[tokio::test]
    async fn head_only_request_returns_metadata_without_bytes() {
        // `options.head = true` short-circuits before any range read:
        // empty payload, the fixed UNIX-epoch last_modified, and a
        // 0..0 range, but a correct `size`.
        let (store, p) = store_with("seg-a.parquet", b"0123456789");
        let res = store
            .get_opts(
                &p,
                GetOptions {
                    head: true,
                    ..Default::default()
                },
            )
            .await
            .expect("head get_opts");
        assert_eq!(res.meta.size, 10);
        assert_eq!(res.meta.last_modified, SUPERFILE_LAST_MODIFIED);
        assert_eq!(res.range, 0..0);
        let bytes = res.bytes().await.expect("bytes");
        assert!(bytes.is_empty(), "HEAD payload carries no bytes");
    }

    #[tokio::test]
    async fn put_multipart_and_copy_are_not_implemented() {
        let (store, p) = store_with("seg-a.parquet", b"abc");
        let mp_err = store
            .put_multipart(&p)
            .await
            .expect_err("multipart upload must fail on the read-only store");
        assert!(matches!(mp_err, OsError::NotImplemented { .. }), "{mp_err}");

        let copy_err = store
            .copy(&p, &ObjPath::from("dst.parquet"))
            .await
            .expect_err("copy must fail on the read-only store");
        assert!(
            matches!(copy_err, OsError::NotImplemented { .. }),
            "{copy_err}"
        );
    }

    #[tokio::test]
    async fn delete_stream_yields_not_implemented_per_location() {
        let (store, p) = store_with("seg-a.parquet", b"abc");
        let locations = stream::iter(vec![Ok(p.clone()), Ok(ObjPath::from("seg-b.parquet"))]);
        let results: Vec<OsResult<ObjPath>> =
            store.delete_stream(locations.boxed()).collect().await;
        assert_eq!(results.len(), 2);
        for r in results {
            let err = r.expect_err("delete must fail on the read-only store");
            assert!(matches!(err, OsError::NotImplemented { .. }), "{err}");
        }
    }

    #[tokio::test]
    async fn list_is_empty_and_list_with_delimiter_returns_no_entries() {
        // The store never advertises its contents during a scan;
        // DataFusion reaches each superfile by its registered path.
        let (store, _p) = store_with("seg-a.parquet", b"abc");
        let listed: Vec<OsResult<ObjectMeta>> = store.list(None).collect().await;
        assert!(listed.is_empty(), "list yields nothing");

        let res = store
            .list_with_delimiter(Some(&ObjPath::from("data")))
            .await
            .expect("list_with_delimiter");
        assert!(res.objects.is_empty());
        assert!(res.common_prefixes.is_empty());
    }

    #[test]
    fn debug_and_display_report_superfile_count() {
        let (store, _p) = store_with("seg-a.parquet", b"abc");
        let dbg = format!("{store:?}");
        assert!(dbg.contains("SuperfileObjectStore"), "{dbg}");
        assert!(dbg.contains("n_superfiles"), "{dbg}");
        let disp = format!("{store}");
        assert_eq!(disp, "SuperfileObjectStore(1 superfiles)");
    }

    #[test]
    fn resolve_range_clamps_every_variant() {
        // Drive the pure range resolver across all four arms,
        // including the past-the-end / oversized clamps.
        assert_eq!(resolve_range(None, 10), 0..10);
        assert_eq!(resolve_range(Some(GetRange::Bounded(2..5)), 10), 2..5);
        // Bounded past the end clamps both ends to `size`.
        assert_eq!(resolve_range(Some(GetRange::Bounded(8..20)), 10), 8..10);
        assert_eq!(resolve_range(Some(GetRange::Offset(7)), 10), 7..10);
        assert_eq!(resolve_range(Some(GetRange::Offset(99)), 10), 10..10);
        assert_eq!(resolve_range(Some(GetRange::Suffix(4)), 10), 6..10);
        assert_eq!(resolve_range(Some(GetRange::Suffix(99)), 10), 0..10);
    }
}
