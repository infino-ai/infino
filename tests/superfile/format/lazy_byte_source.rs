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

    let source = BytesLazyByteSource::new(bytes);
    let lazy = SuperfileReader::open_lazy(&source)
        .await
        .expect("lazy open");

    assert_eq!(lazy.schema(), eager.schema());
    assert_eq!(lazy.id_column(), eager.id_column());
    assert_eq!(lazy.n_docs(), eager.n_docs());
    assert_eq!(lazy.fts_columns(), eager.fts_columns());

    // FTS terms identical between the two readers.
    let lazy_terms = lazy.fts().expect("fts").iter_column_terms("title");
    let eager_terms = eager.fts().expect("fts").iter_column_terms("title");
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

#[tokio::test]
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

    let source = StorageRangeSource::new(Arc::clone(&proxy) as Arc<dyn StorageProvider>, uri)
        .await
        .expect("source");
    let head_after_construct = proxy.head_calls.load(Ordering::Acquire);
    assert_eq!(
        head_after_construct, 1,
        "StorageRangeSource::new must HEAD the object once"
    );

    let reader = SuperfileReader::open_lazy(&source)
        .await
        .expect("open_lazy");
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

#[tokio::test]
async fn open_lazy_via_storage_matches_open_via_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = "data/seg-equiv.sf";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let eager = SuperfileReader::open(bytes).expect("eager");
    let source = StorageRangeSource::new(Arc::clone(&local), uri)
        .await
        .expect("source");
    let lazy = SuperfileReader::open_lazy(&source).await.expect("lazy");

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
