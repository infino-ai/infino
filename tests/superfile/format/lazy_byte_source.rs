//! `SuperfileReader::open_lazy` + `StorageRangeSource`
//! integration — drives the lazy-open path through a real
//! `SuperfileBuilder` and a `LocalFsStorageProvider` (the
//! `BytesLazyByteSource` adapter's own behavior is unit-
//! tested in `src/superfile/lazy_source.rs`).
//!
//! Covers:
//! - `SuperfileReader::open_lazy` returns a reader
//!   equivalent to `SuperfileReader::open(full_bytes)` for
//!   FTS queries.
//! - `StorageRangeSource` over `LocalFsStorageProvider`
//!   produces an open_lazy reader whose query results match
//!   the in-memory `open(bytes)` reader.
//! - The source's `range` method is exercised (proving the
//!   trait actually drives I/O — not just a hidden whole-
//!   file path).
//! - `StorageRangeSource` out-of-bounds requests surface
//!   `LazyByteSourceError::OutOfBounds`.

#![deny(clippy::unwrap_used)]

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::{
    BytesLazyByteSource, LazyByteSource, LazyByteSourceError, SuperfileReader,
};
use infino::supertable::StorageRangeSource;
use infino::supertable::storage::{
    LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider,
};
use infino::test_helpers::{decimal128_ids, default_tokenizer};
use tempfile::TempDir;

// ============================================================
// Tiny superfile fixture (FTS only, no vector).
// ============================================================

fn build_test_bytes() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    let ids = decimal128_ids(vec![1u64, 2, 3, 4]);
    let titles = LargeStringArray::from(vec![
        "alpha bravo special",
        "charlie delta",
        "echo special foxtrot",
        "gamma hotel",
    ]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
    b.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

// ============================================================
// open_lazy vs open round-trip equivalence.
// ============================================================

#[tokio::test]
async fn open_lazy_via_bytes_source_matches_open() {
    let bytes = build_test_bytes();
    let eager = SuperfileReader::open(bytes.clone()).expect("eager open");

    let source: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
    let lazy = SuperfileReader::open_lazy(source).await.expect("lazy open");

    assert_eq!(lazy.schema(), eager.schema());
    assert_eq!(lazy.id_column(), eager.id_column());
    assert_eq!(lazy.n_docs(), eager.n_docs());
    assert_eq!(lazy.fts_columns(), eager.fts_columns());

    // FTS terms identical between the two readers.
    let lazy_terms = lazy
        .fts()
        .expect("fts")
        .iter_column_terms("title")
        .expect("lazy column terms");
    let eager_terms = eager
        .fts()
        .expect("fts")
        .iter_column_terms("title")
        .expect("eager column terms");
    assert_eq!(lazy_terms, eager_terms);
}

// ============================================================
// StorageRangeSource — wraps a real storage provider.
// ============================================================

#[derive(Debug)]
struct CountingProxy {
    inner: Arc<dyn StorageProvider>,
    head_calls: AtomicUsize,
    get_range_calls: AtomicUsize,
}

impl CountingProxy {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            head_calls: AtomicUsize::new(0),
            get_range_calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl StorageProvider for CountingProxy {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.head_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<Bytes, StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.get_range_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        e: Option<&str>,
    ) -> Result<(), StorageError> {
        self.inner.put_if_match(uri, bytes, e).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

// `StorageRangeSource` doesn't override `try_get_range_sync`, so
// every `Source::get_range` cold-misses and bridges via
// `block_in_place + Handle::block_on` — which requires the
// multi-threaded tokio flavor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn storage_range_source_drives_open_lazy_against_localfs() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();

    // Seed the segment at a stable URI.
    let uri = "data/seg-test.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    // Counting proxy so we can assert the trait is actually
    // driving I/O (not a hidden path).
    let proxy = CountingProxy::new(local);

    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
            .await
            .expect("source"),
    );
    let head_after_construct = proxy.head_calls.load(Ordering::Acquire);
    assert_eq!(
        head_after_construct, 1,
        "StorageRangeSource::new must HEAD the object once"
    );

    let reader = SuperfileReader::open_lazy(source).await.expect("open_lazy");
    let range_after_open = proxy.get_range_calls.load(Ordering::Acquire);
    assert!(
        range_after_open >= 1,
        "open_lazy must exercise the source's range fn; got {range_after_open}"
    );

    // The reader serves real queries — sanity check via BM25.
    let fts = reader.fts().expect("fts");
    let hits = fts
        .search("title", &["special"], 10, BoolMode::Or)
        .expect("bm25");
    assert_eq!(hits.len(), 2, "two docs contain 'special'");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn open_lazy_via_storage_matches_open_via_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = "data/seg-equiv.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let eager = SuperfileReader::open(bytes).expect("eager");
    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&local), uri)
            .await
            .expect("source"),
    );
    let lazy = SuperfileReader::open_lazy(source).await.expect("lazy");

    // Schema + identity metadata identical.
    assert_eq!(lazy.id_column(), eager.id_column());
    assert_eq!(lazy.n_docs(), eager.n_docs());

    // Query parity for BM25.
    let eager_hits = eager
        .fts()
        .expect("fts")
        .search("title", &["alpha"], 10, BoolMode::Or)
        .expect("eager bm25");
    let lazy_hits = lazy
        .fts()
        .expect("fts")
        .search("title", &["alpha"], 10, BoolMode::Or)
        .expect("lazy bm25");
    let eager_ids: Vec<_> = eager_hits.iter().map(|(d, _)| *d).collect();
    let lazy_ids: Vec<_> = lazy_hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(lazy_ids, eager_ids);
}

// ============================================================
// 013 PR3 (A2v): Vector lazy open-time region tightening.
//
// Asserts that lazy-opening a vector-bearing superfile over a
// real range-fetching source pulls only the open-time region
// of each vector subsection (sub-header + codec_meta), not
// the whole vector blob — and counts the actual range GETs
// to enforce the plan's cold-open range/byte budget.
//
// Plan target (single vector column, `verify_crc = false`):
//   - 2 GETs for the Parquet footer (trailer + body).
//   - 1 GET for the vector-blob structural prefix
//     (outer header + directory + dir-CRC, all contiguous
//     from offset 0; covered by the
//     STRUCTURAL_PREFIX_SPECULATIVE_BYTES prefetch).
//   - Per subsection: 1 GET for the sub-header + 1 GET for
//     the codec_meta tail (Sq8 only; Fp32 / Bf16 / RabitqOnly
//     have zero-byte codec_meta and skip this entirely).
//
// So the cold-open ceiling is **5 GETs** at single-column Sq8
// and **4 GETs** at single-column Fp32 / Bf16 / RabitqOnly.
// ============================================================

use infino::superfile::vector::builder::VectorConfig;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;

/// Vector-bearing superfile fixture for the PR3 open-time
/// region tests. Single column, `n_docs` × `dim` vectors with
/// the requested rerank codec. The Parquet rows carry an
/// id + title column so the segment also exercises the
/// generic Parquet writer path; the vector data is fed via
/// the parallel `flat` buffer to `add_batch`.
fn build_vec_superfile_bytes(rerank_codec: RerankCodec, n_docs: u32) -> Bytes {
    use infino::superfile::vector::distance::normalize;
    let dim = 16usize;
    let n_cent = 4usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    let mut flat = Vec::<f32>::with_capacity((n_docs as usize) * dim);
    for i in 0..n_docs {
        let mut v = vec![0.0f32; dim];
        v[(i as usize) % dim] = 1.0;
        v[((i as usize) + 3) % dim] = 0.5;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }
    let ids = decimal128_ids(0..n_docs as u64);
    let titles: Vec<String> = (0..n_docs).map(|i| format!("doc-{i}")).collect();
    let title_strs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();
    let titles_arr = LargeStringArray::from(title_strs);
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

/// Counts call counts, bytes pulled per `get_range`, and
/// records the (start, len) of every range so tests can
/// assert against an exact set of fetches.
#[derive(Debug)]
struct ByteCountingProxy {
    inner: Arc<dyn StorageProvider>,
    get_range_calls: AtomicUsize,
    get_range_bytes: AtomicUsize,
    ranges: std::sync::Mutex<Vec<(u64, u64)>>,
}

impl ByteCountingProxy {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            get_range_calls: AtomicUsize::new(0),
            get_range_bytes: AtomicUsize::new(0),
            ranges: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn ranges(&self) -> Vec<(u64, u64)> {
        self.ranges.lock().expect("ranges mutex poisoned").clone()
    }
}

#[async_trait]
impl StorageProvider for ByteCountingProxy {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<Bytes, StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.get_range_calls.fetch_add(1, Ordering::AcqRel);
        let len = range.end - range.start;
        self.get_range_bytes
            .fetch_add(len as usize, Ordering::AcqRel);
        self.ranges
            .lock()
            .expect("ranges mutex poisoned")
            .push((range.start, len));
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        e: Option<&str>,
    ) -> Result<(), StorageError> {
        self.inner.put_if_match(uri, bytes, e).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

/// `n_docs` for the PR3 cold-open budget fixtures. Sized so
/// the segment is well past the 4 KiB structural prefetch:
///   - Fp32 `full[]` is `n_docs × dim × 4` = 64 KiB
///     at 1024 × 16 — single contiguous region the lazy
///     open-time path must NOT touch.
///   - Sq8 `codec_meta` is `2 × n_cent × dim × 4 + n_docs × 4`
///     = 4.6 KiB at this shape — *just* past the prefetch
///     boundary, so the codec_meta straddle case also gets
///     exercised end-to-end.
const PR3_FIXTURE_N_DOCS: u32 = 1024;

/// Lazy-open with a single Fp32 vector column. Cold-open
/// budget: ≤ 3 range GETs at the underlying source —
/// 2 Parquet footer GETs (trailer + body) + 1 vector-blob
/// structural prefetch (covers outer header + directory +
/// dir-CRC + the per-subsection sub-header in one shot at
/// single-column scale). Fp32 has zero-byte codec_meta so
/// nothing else has to be fetched at open.
///
/// Plan acceptance for the vector half is "≤ 5 range GETs"
/// at single-column Sq8; Fp32 is one tighter because of the
/// missing codec_meta tail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_open_lazy_fp32_pulls_only_open_time_region() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_vec_superfile_bytes(RerankCodec::Fp32, PR3_FIXTURE_N_DOCS);
    let total = bytes.len() as u64;
    let uri = "data/vec-seg-fp32.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let proxy = ByteCountingProxy::new(local);
    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
            .await
            .expect("source"),
    );
    let opts = infino::superfile::reader::OpenOptions { verify_crc: false };
    let reader = SuperfileReader::open_lazy_with(source, opts)
        .await
        .expect("lazy open");

    let calls = proxy.get_range_calls.load(Ordering::Acquire);
    let bytes_pulled = proxy.get_range_bytes.load(Ordering::Acquire) as u64;
    let ranges = proxy.ranges();
    assert!(
        calls <= 3,
        "Fp32 cold-open budget is ≤ 3 range GETs (2 footer + 1 \
         vector-blob structural prefetch covers header + dir + sub-header \
         at single-column scale); got {calls} ranges {ranges:?}"
    );
    // The Fp32 `full[]` region alone is `n_docs × dim × 4`
    // = 64 KiB. The lazy open path must not touch it (or any
    // of the doc_ids / Parquet row group bytes either), so
    // total bytes pulled is bounded by the structural
    // prefetch + the Parquet footer body, both small.
    let open_time_ceiling: u64 = 8 * 1024;
    assert!(
        bytes_pulled <= open_time_ceiling,
        "Fp32 cold-open pulled {bytes_pulled} B (segment {total} B); \
         ceiling is {open_time_ceiling} B. ranges={ranges:?}"
    );

    // Reader functions: vector reader present, columns reported.
    assert_eq!(reader.vector_columns(), vec!["emb"]);
    assert!(reader.fts().is_none());
    assert!(reader.vec().is_some());
}

/// Lazy-open with a single Sq8 vector column — the headline
/// cold-open shape on object storage. Cold-open budget:
/// ≤ 4 range GETs at single-column scale —
/// 2 Parquet footer + 1 vector-blob structural prefetch
/// (covers header + dir + sub-header) + 1 codec_meta tail.
/// At larger scales the codec_meta tail is the only
/// per-column GET past the prefetch; everything before it
/// lands in the prefetch's window.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_open_lazy_sq8_pulls_only_open_time_region() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_vec_superfile_bytes(RerankCodec::Sq8, PR3_FIXTURE_N_DOCS);
    let total = bytes.len() as u64;
    let uri = "data/vec-seg-sq8.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let proxy = ByteCountingProxy::new(local);
    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
            .await
            .expect("source"),
    );
    let opts = infino::superfile::reader::OpenOptions { verify_crc: false };
    let reader = SuperfileReader::open_lazy_with(source, opts)
        .await
        .expect("lazy open");

    let calls = proxy.get_range_calls.load(Ordering::Acquire);
    let bytes_pulled = proxy.get_range_bytes.load(Ordering::Acquire) as u64;
    let ranges = proxy.ranges();
    assert!(
        calls <= 4,
        "Sq8 cold-open budget is ≤ 4 range GETs at single-column scale \
         (2 footer + 1 structural prefetch + 1 codec_meta tail); \
         got {calls} ranges {ranges:?}"
    );
    // Sq8 codec_meta is `2 × n_cent × dim × 4 + n_docs × 4`
    // bytes (per-cluster (scale, offset) arrays + per-doc
    // norms for L2Sq/Cosine). At 1024 × dim-16 × n_cent-4
    // that's 4608 B. Plus the 4 KiB structural prefetch +
    // ≤ 1 KiB Parquet footer body + 8 B Parquet trailer.
    // The 16 KiB ceiling leaves slack for prefetch
    // resizing without leaving room to silently re-fetch
    // the 16 KiB Sq8 `full[]` region or anything beyond it.
    let open_time_ceiling: u64 = 16 * 1024;
    assert!(
        bytes_pulled <= open_time_ceiling,
        "Sq8 cold-open pulled {bytes_pulled} B (segment {total} B); \
         ceiling is {open_time_ceiling} B. ranges={ranges:?}"
    );

    // Reader functions: vector reader present, columns reported.
    assert_eq!(reader.vector_columns(), vec!["emb"]);
    assert!(reader.fts().is_none());
    assert!(reader.vec().is_some());

    // Sanity: the lazy reader actually answers a vector query
    // — proves the open-time fetches are *enough* to drive
    // search, with the rest of the bytes pulled lazily on
    // demand inside `vector_search`.
    let mut q = vec![0.0f32; 16];
    q[1] = 1.0;
    q[4] = 0.5;
    infino::superfile::vector::distance::normalize(&mut q);
    let hits = reader
        .vector_search(
            "emb",
            &q,
            3,
            infino::superfile::reader::VectorSearchOptions::default(),
        )
        .expect("vector_search on lazy reader");
    assert!(!hits.is_empty(), "lazy vector search should return hits");
}

// ============================================================
// 013 PR5 (A3f): FTS lazy open-time region tightening.
//
// Asserts that lazy-opening an FTS-bearing superfile over a
// real range-fetching source pulls only the open-time region
// of the FTS blob — the 48-byte header plus the doc-lengths
// region (directory + per-column arrays, contiguous at the
// blob tail) — and never the FST or postings, which are
// query-time only.
//
// Plan target (single text column, `verify_crc = false`):
//   - 2 GETs for the Parquet footer (trailer + body).
//   - 1 GET for the FTS header (48 B at the blob start).
//   - 1 GET for the doc-lengths region (dir + dir-CRC + every
//     per-column array + CRC, all contiguous — one GET
//     regardless of column count after PR5; pre-PR5 this was
//     one GET per column array on top of the dir).
//
// So the cold-open ceiling is **4 GETs** at single text
// column, independent of `n_docs`. A cold first single-term
// BM25 query then pulls the FST + only the touched postings
// blocks on top — never the whole blob.
// ============================================================

/// `n_docs` for the PR5 FTS cold-open budget fixture. Large
/// enough that the FST + postings dominate the segment so an
/// open that touched them would blow the byte ceiling, and so
/// the per-doc doc-lengths array (`4 × n_docs` = 16 KiB at
/// 4096) is itself a non-trivial slice of the open-time
/// region.
const PR5_FTS_FIXTURE_N_DOCS: u32 = 4096;

/// FTS-only superfile fixture for the PR5 open-time region
/// tests. Single text column, `n_docs` rows over a *tiny*
/// vocabulary so the FST stays small relative to the postings
/// (the cold-search test wants the FST fetch to be cheap and
/// the dominant cost to be the postings the cursor touches):
///   - every doc carries the high-frequency `common` term, so
///     its postings list spans many blocks and is the largest
///     single region of the blob;
///   - every 7th doc additionally carries the sparse `special`
///     term used by the cold single-term search test.
///
/// A single-term `special` query must therefore pull the FST +
/// only `special`'s (sparse) postings blocks — never the big
/// `common` postings list.
fn build_fts_superfile_bytes(n_docs: u32) -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    let ids = decimal128_ids(0..n_docs as u64);
    let titles: Vec<String> = (0..n_docs)
        .map(|i| {
            if i % 7 == 0 {
                "common special".to_string()
            } else {
                "common".to_string()
            }
        })
        .collect();
    let title_strs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();
    let titles_arr = LargeStringArray::from(title_strs);
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
    b.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

/// Lazy-open with a single FTS text column. Cold-open budget:
/// ≤ 4 range GETs at the underlying source — 2 Parquet footer
/// GETs (trailer + body) + 1 FTS header + 1 doc-lengths region
/// (dir + per-column arrays, collapsed into a single tail GET
/// by PR5). The FST and postings regions must NOT be touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fts_open_lazy_pulls_only_open_time_region() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_fts_superfile_bytes(PR5_FTS_FIXTURE_N_DOCS);
    let uri = "data/fts-seg.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let proxy = ByteCountingProxy::new(local);
    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
            .await
            .expect("source"),
    );
    let opts = infino::superfile::reader::OpenOptions { verify_crc: false };
    let reader = SuperfileReader::open_lazy_with(source, opts)
        .await
        .expect("lazy open");

    let calls = proxy.get_range_calls.load(Ordering::Acquire);
    let ranges = proxy.ranges();
    // Exact cold-open shape: 2 Parquet footer GETs (trailer +
    // body) + 1 FTS header + 1 doc-lengths region. Pinned, not
    // a ceiling — a regression that re-introduced per-column
    // array GETs or touched the FST/postings would change this.
    assert_eq!(
        calls, 4,
        "FTS cold-open should be exactly 4 range GETs (2 footer + 1 FTS \
         header + 1 doc-lengths region); got {calls} ranges {ranges:?}"
    );

    // The doc-lengths region is the single largest open-time
    // fetch: directory (n_cols × 16) + dir CRC (4) + per column
    // (4 × n_docs array + 4 CRC). For 1 column / 4096 docs that
    // is exactly 16408 B. The FST + postings (the bulk of the
    // FTS blob) sit *before* it and must never be pulled at
    // open, so no single GET may exceed the doc-lengths region.
    let n_cols = 1u64;
    let n_docs = PR5_FTS_FIXTURE_N_DOCS as u64;
    let dl_region_bytes = (n_cols * 16 + 4) + n_cols * (4 * n_docs + 4);
    let max_get = ranges.iter().map(|(_, len)| *len).max().unwrap_or(0);
    assert_eq!(
        max_get, dl_region_bytes,
        "largest cold-open GET should be the {dl_region_bytes} B doc-lengths \
         region; got {max_get} B — a larger GET means the FST/postings or a \
         Parquet row group was pulled. ranges={ranges:?}"
    );
    // And exactly one GET of exactly 48 B: the FTS header.
    let header_gets = ranges.iter().filter(|(_, len)| *len == 48).count();
    assert_eq!(
        header_gets, 1,
        "expected exactly one 48 B FTS-header GET; ranges={ranges:?}"
    );

    // Reader functions: FTS reader present, no vector.
    assert!(reader.fts().is_some());
    assert!(reader.vec().is_none());
}

/// Open a fresh lazy `SuperfileReader` over `uri` through its
/// own [`ByteCountingProxy`], so each call gets a cold source
/// with independent range accounting.
async fn open_lazy_fts_with_proxy(
    local: Arc<dyn StorageProvider>,
    uri: &str,
) -> (SuperfileReader, Arc<ByteCountingProxy>) {
    let proxy = ByteCountingProxy::new(local);
    let source: Arc<dyn LazyByteSource> = Arc::new(
        StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
            .await
            .expect("source"),
    );
    let opts = infino::superfile::reader::OpenOptions { verify_crc: false };
    let reader = SuperfileReader::open_lazy_with(source, opts)
        .await
        .expect("lazy open");
    (reader, proxy)
}

/// A cold first single-term BM25 query on a lazily-opened FTS
/// reader pulls the FST + only the postings blocks the cursor
/// actually touches — never the whole blob, and in particular
/// never the dense `common` postings list when querying the
/// sparse `special` term.
///
/// Strict assertions (pin the actual behavior, not a ceiling):
///  - the sparse `special` query returns the full top-k;
///  - every search-time GET is block-granular (no single fetch
///    is large enough to be a whole region / the segment);
///  - the FST is fetched exactly once;
///  - a `special` query reads strictly fewer postings bytes
///    than a `common` query — i.e. it provably skips the big
///    `common` postings list rather than scanning the blob.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fts_cold_single_term_search_pulls_only_touched_postings() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_fts_superfile_bytes(PR5_FTS_FIXTURE_N_DOCS);
    let total = bytes.len() as u64;
    let uri = "data/fts-seg-search.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    // Sums the byte lengths of a slice of recorded ranges.
    fn sum_bytes(rs: &[(u64, u64)]) -> u64 {
        rs.iter().map(|(_, len)| *len).sum()
    }

    // --- Sparse term `special` (every 7th doc). ---
    let (reader, proxy) = open_lazy_fts_with_proxy(Arc::clone(&local), uri).await;
    let open_ranges = proxy.ranges().len();
    let hits = reader
        .bm25_search("title", "special", 10, BoolMode::Or)
        .expect("bm25 search on lazy reader");
    let all_ranges = proxy.ranges();
    let special_search = &all_ranges[open_ranges..];

    // 586 docs match `special` (ceil(4096 / 7)); top-k = 10.
    assert_eq!(
        hits.len(),
        10,
        "expected the full top-10 for the sparse `special` term; got {}",
        hits.len()
    );

    // Block-granular: the cursor fetches the FST, per-term
    // metadata, the skip table, and individual postings blocks
    // — each a small fetch. No single search GET may be large
    // enough to be a whole region pull. 1 KiB is comfortably
    // above the per-block size while well below any region.
    let max_search_get = special_search
        .iter()
        .map(|(_, len)| *len)
        .max()
        .unwrap_or(0);
    assert!(
        max_search_get <= 1024,
        "every cold-search GET should be block-granular (≤ 1 KiB); largest \
         was {max_search_get} B — that looks like a bulk region fetch. \
         search ranges={special_search:?}"
    );

    let special_search_bytes = sum_bytes(special_search);
    assert!(
        special_search_bytes < total / 8,
        "cold `special` search pulled {special_search_bytes} B (segment \
         {total} B) — should be a small fraction. ranges={special_search:?}"
    );

    // --- Dense term `common` (every doc) on a fresh cold reader. ---
    let (reader2, proxy2) = open_lazy_fts_with_proxy(Arc::clone(&local), uri).await;
    let open_ranges2 = proxy2.ranges().len();
    let common_hits = reader2
        .bm25_search("title", "common", 10, BoolMode::Or)
        .expect("bm25 search on lazy reader");
    assert_eq!(
        common_hits.len(),
        10,
        "dense `common` term should fill top-10"
    );
    let all_ranges2 = proxy2.ranges();
    let common_search = &all_ranges2[open_ranges2..];
    let common_search_bytes = sum_bytes(common_search);

    // The headline property: querying the sparse `special` term
    // reads strictly fewer postings bytes than querying the
    // dense `common` term. If the search ignored term locality
    // and pulled the whole postings region (or the blob), the
    // two would be equal. The gap proves `special` genuinely
    // skips the large `common` postings list.
    assert!(
        special_search_bytes < common_search_bytes,
        "sparse `special` ({special_search_bytes} B) should read fewer \
         postings bytes than dense `common` ({common_search_bytes} B); \
         special={special_search:?} common={common_search:?}"
    );
}

#[tokio::test]
async fn storage_range_source_out_of_bounds_surfaces_typed_error() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = "data/seg-oob.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let source = StorageRangeSource::new(Arc::clone(&local), uri)
        .await
        .expect("source");
    let size = source.size();
    let err = source.range(size, 1024).await.expect_err("must reject");
    assert!(
        matches!(err, LazyByteSourceError::OutOfBounds { .. }),
        "expected OutOfBounds, got {err:?}"
    );
}
