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
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use object_store::{
    Attributes, CopyOptions, Error as OsError, GetOptions, GetRange, GetResult, GetResultPayload,
    ListResult, MultipartUpload, OBJECT_STORE_COALESCE_DEFAULT, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OsResult, coalesce_ranges,
    path::Path as ObjPath,
};

use crate::{
    runtime_metrics::op_stats::{self, OpStatsCollector},
    superfile::{CacheBlocks, CacheRoom, LazyByteSource, lazy_source::Source},
    supertable::error::QueryError,
};

/// Fixed `last_modified` reported for every registered superfile.
/// Superfiles are immutable once committed, so a wall-clock timestamp
/// carries no signal here — and a value that changed on every call
/// would defeat any downstream cache keyed on `(path, last_modified)`
/// and make responses non-deterministic.
const SUPERFILE_LAST_MODIFIED: DateTime<Utc> = DateTime::UNIX_EPOCH;

/// Reads [`SuperfileObjectStore::get_spans_ahead`] keeps in flight at once,
/// the same as object_store's own `get_ranges` (which keeps it private).
const FETCH_PARALLELISM: usize = 10;

/// Read-only [`ObjectStore`] backed by per-superfile [`LazyByteSource`]s.
///
/// A pinned provider constructs one registry with [`Self::new`], fills it via
/// [`Self::insert_source`] as immutable files are first opened, and reuses it
/// for every DataFusion scan of that manifest.
pub(crate) struct SuperfileObjectStore {
    /// Per-query work collector, captured at construction. On the catalog
    /// path (`Connection::query_sql`) the provider — and therefore this
    /// store — is built fresh per query on the caller's thread, so the
    /// capture is per-query by construction and can never leak across
    /// queries. The reader-level cached `SessionContext`
    /// (`sql_session_context`) instead builds its provider under
    /// `op_stats::suppressed`, so this is `None` there by design — a
    /// collector riding a cached context would bill later queries into
    /// an old scope. That cached path serves mutation id-capture
    /// (`scan_ids_matching`) and the test-only reader `query_sql`,
    /// neither of which is part of the metered read surface.
    op_stats: Option<Arc<OpStatsCollector>>,
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
            op_stats: op_stats::current(),
            sources: Arc::new(DashMap::new()),
        }
    }

    /// The per-query collector this store captured at construction — the
    /// provider flushes its predicate-walk work through the same channel
    /// that meters this scan's page bytes.
    pub(crate) fn op_stats(&self) -> Option<Arc<OpStatsCollector>> {
        self.op_stats.clone()
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

    /// How the cache in front of `location` sizes a miss's download (see
    /// [`LazyByteSource::cache_blocks`]). `None` if nothing caches it.
    pub(crate) fn cache_blocks(&self, location: &ObjPath) -> Option<CacheBlocks> {
        self.source(location).ok()?.cache_blocks()
    }

    /// Reserve room in the cache in front of `location` for `bytes` downloaded
    /// ahead (see [`LazyByteSource::cache_room_ahead`]).
    pub(crate) fn cache_room_ahead(&self, location: &ObjPath, bytes: u64) -> Option<CacheRoom> {
        self.source(location).ok()?.cache_room_ahead(bytes)
    }

    /// Count `ranges` of `location` as read, for a reader served them from
    /// bytes it downloaded ahead. See [`LazyByteSource::note_read`].
    pub(crate) fn note_reads(&self, location: &ObjPath, ranges: &[Range<u64>]) {
        if let Ok(source) = self.source(location) {
            for range in ranges {
                source.note_read(range.start, range.end - range.start);
            }
        }
    }

    /// Whether all of `range` in `location` can be read without a download.
    /// See [`LazyByteSource::is_local`]: it copies nothing and isn't counted
    /// as a read.
    pub(crate) fn is_local(&self, location: &ObjPath, range: &Range<u64>) -> bool {
        range.is_empty()
            || self
                .source(location)
                .is_ok_and(|source| source.is_local(range.start, range.end - range.start))
    }

    /// Fetch `ranges` the way object_store's own `get_ranges` does
    /// ([`coalesce_ranges`]), without counting them as page reads. Callers
    /// count first: [`ObjectStore::get_ranges`], and read-ahead for the part
    /// of a request a batch didn't cover.
    pub(crate) async fn get_ranges_uncounted(
        &self,
        location: &ObjPath,
        ranges: &[Range<u64>],
    ) -> OsResult<Vec<Bytes>> {
        let source = self.source(location)?;
        coalesce_ranges(
            ranges,
            |range| read_range(Arc::clone(&source), range),
            OBJECT_STORE_COALESCE_DEFAULT,
        )
        .await
    }

    /// Download `clusters` of spans ahead of the scan, through
    /// [`LazyByteSource::range_ahead`], and return every span's bytes in
    /// order. Not counted as page reads.
    /// - Clusters download [`FETCH_PARALLELISM`] at a time.
    /// - A cluster's spans download one after another, so a cache block they
    ///   share is downloaded once.
    pub(crate) async fn get_spans_ahead(
        &self,
        location: &ObjPath,
        clusters: &[Vec<Range<u64>>],
    ) -> OsResult<Vec<Bytes>> {
        let source = self.source(location)?;
        let per_cluster: Vec<Vec<Bytes>> = stream::iter(clusters.iter().cloned())
            .map(|cluster| {
                let source = Arc::clone(&source);
                async move {
                    let mut bytes = Vec::with_capacity(cluster.len());
                    for span in cluster {
                        bytes.push(read_range_ahead(&source, span).await?);
                    }
                    Ok::<_, OsError>(bytes)
                }
            })
            .buffered(FETCH_PARALLELISM)
            .try_collect()
            .await?;
        Ok(per_cluster.into_iter().flatten().collect())
    }

    /// `ranges` clamped to the object at `location`, as object_store's
    /// default path clamps each range in `get_opts`: a range past the end
    /// reads short, not fails.
    pub(crate) fn clamped(
        &self,
        location: &ObjPath,
        ranges: &[Range<u64>],
    ) -> OsResult<Vec<Range<u64>>> {
        let size = self.source(location)?.size();
        Ok(ranges
            .iter()
            .map(|r| r.start.min(size)..r.end.min(size))
            .collect())
    }

    /// Count `ranges` as page reads: one per merged range, gaps included, as
    /// object_store's default `get_ranges` counted them through `get_opts`.
    /// [`ObjectStore::get_ranges`] counts through here, so a request served
    /// from bytes read ahead counts the same as one fetched.
    pub(crate) fn count_page_reads(&self, ranges: &[Range<u64>]) {
        for range in merged_ranges(ranges) {
            self.count_page_read(range.end - range.start);
        }
    }

    fn count_page_read(&self, len: u64) {
        if let Some(stats) = &self.op_stats
            && len > 0
        {
            // Parquet-driven plan reads: page/footer ranges the scan needs,
            // counted before the byte source decides resident vs fetch.
            stats.add_sql_page_bytes(len);
            stats.add_planned_read_ranges(1);
        }
    }

    fn source(&self, location: &ObjPath) -> OsResult<Arc<dyn LazyByteSource>> {
        self.sources
            .get(location)
            .map(|source| Arc::clone(source.value()))
            .ok_or_else(|| OsError::NotFound {
                path: location.to_string(),
                // Every superfile a scan reads is registered before it runs,
                // so a miss is our bug, not a missing object: the source says
                // so, and the query's error reports an engine fault.
                source: Box::new(QueryError::Internal(format!(
                    "superfile {location} not registered in SuperfileObjectStore"
                ))),
            })
    }
}

/// `range` of `source`, downloaded ahead of the scan (see
/// [`LazyByteSource::range_ahead`]).
async fn read_range_ahead(source: &Arc<dyn LazyByteSource>, range: Range<u64>) -> OsResult<Bytes> {
    if range.is_empty() {
        return Ok(Bytes::new());
    }
    source
        .range_ahead(range.start, range.end - range.start)
        .await
        .map_err(|e| OsError::Generic {
            store: "SuperfileObjectStore",
            source: Box::new(e),
        })
}

/// `range` of `source`. Goes through `range_async` so a warm, mmap/block-
/// resident scan serves its bytes synchronously (zero-copy, no I/O) instead
/// of awaiting a range fetch; a cold miss still `await`s the GET.
async fn read_range(source: Arc<dyn LazyByteSource>, range: Range<u64>) -> OsResult<Bytes> {
    if range.is_empty() {
        return Ok(Bytes::new());
    }
    Source::Lazy(source)
        .range_async(range.start as usize..range.end as usize)
        .await
        .map_err(|e| OsError::Generic {
            store: "SuperfileObjectStore",
            source: Box::new(e),
        })
}

/// How the store merges `ranges` into reads: sorted, and merged where the
/// gap between two is at most object_store's default coalesce distance. The
/// one merge rule for counting and charging, and for fetching, except that a
/// scan's own reads may also join cold pages of one column chunk up to 2 MiB
/// apart (see the read-ahead module). It matches
/// object_store's own `coalesce_ranges` (pinned by a test), so merging is the
/// same as the default `get_ranges`.
pub(crate) fn merged_ranges(ranges: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match merged.last_mut() {
            Some(last)
                if range
                    .start
                    .checked_sub(last.end)
                    .is_none_or(|gap| gap <= OBJECT_STORE_COALESCE_DEFAULT) =>
            {
                last.end = last.end.max(range.end);
            }
            _ => merged.push(range),
        }
    }
    merged
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
        self.count_page_read(range.end.saturating_sub(range.start));
        let bytes = read_range(source, range.clone()).await?;

        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes: Attributes::default(),
        })
    }

    async fn get_ranges(&self, location: &ObjPath, ranges: &[Range<u64>]) -> OsResult<Vec<Bytes>> {
        let ranges = self.clamped(location, ranges)?;
        self.count_page_reads(&ranges);
        self.get_ranges_uncounted(location, &ranges).await
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
    use std::{slice, sync::Mutex};

    use object_store::ObjectStoreExt;

    use super::*;
    use crate::superfile::{BytesLazyByteSource, LazyByteSourceError};

    fn store_with(path: &str, body: &'static [u8]) -> (SuperfileObjectStore, ObjPath) {
        let p = ObjPath::from(path);
        let mut sources: HashMap<ObjPath, Arc<dyn LazyByteSource>> = HashMap::new();
        sources.insert(
            p.clone(),
            Arc::new(BytesLazyByteSource::new(Bytes::from_static(body))),
        );
        (SuperfileObjectStore::from_sources(sources), p)
    }

    /// Returns half of every range it's asked for, like an object replaced by
    /// a shorter one.
    #[derive(Debug)]
    struct ShortSource;

    #[async_trait]
    impl LazyByteSource for ShortSource {
        fn size(&self) -> u64 {
            SHORT_SOURCE_SIZE
        }

        async fn range(&self, _start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            Ok(Bytes::from(vec![0u8; (len / 2) as usize]))
        }
    }

    /// The size [`ShortSource`] claims.
    const SHORT_SOURCE_SIZE: u64 = 100;

    #[tokio::test]
    async fn a_short_read_comes_back_short() {
        // A source that returns fewer bytes than asked:
        //  - the store hands the short bytes on, as object_store's `get_ranges` does;
        //  - the Parquet decoder then reports it, instead of the store panicking.
        // This test pins the short read reaching the caller.
        let p = ObjPath::from("short.parquet");
        let mut sources: HashMap<ObjPath, Arc<dyn LazyByteSource>> = HashMap::new();
        sources.insert(p.clone(), Arc::new(ShortSource));
        let store = SuperfileObjectStore::from_sources(sources);

        let wanted: Range<u64> = 10..40;
        let got = store
            .get_ranges(&p, slice::from_ref(&wanted))
            .await
            .expect("ranges");
        let asked = wanted.end - wanted.start;
        assert_eq!(got[0].len() as u64, asked / 2, "half of what was asked");
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

    #[tokio::test]
    async fn merged_ranges_are_what_a_fetch_fetches() {
        const GAP: u64 = OBJECT_STORE_COALESCE_DEFAULT;
        let cases: [&[Range<u64>]; 7] = [
            &[],
            &[0..10, 5..20],
            &[0..10, 10..20],
            &[0..10, 10 + GAP..20 + GAP],
            &[0..10, 11 + GAP..20 + GAP],
            &[100..200, 0..10, 50..60],
            &[0..10, 0..0, 3..4],
        ];
        for ranges in cases {
            let fetched = Mutex::new(Vec::new());
            coalesce_ranges(
                ranges,
                |range: Range<u64>| {
                    fetched.lock().expect("fetched").push(range.clone());
                    async move {
                        Ok::<_, OsError>(Bytes::from(vec![0; (range.end - range.start) as usize]))
                    }
                },
                GAP,
            )
            .await
            .expect("coalesce");
            let mut fetched = fetched.into_inner().expect("fetched");
            fetched.sort_unstable_by_key(|range| range.start);
            fetched.retain(|range| !range.is_empty());
            let mut merged = merged_ranges(ranges);
            merged.retain(|range| !range.is_empty());
            assert_eq!(merged, fetched, "{ranges:?}");
        }
    }
}
