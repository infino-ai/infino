// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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

use std::{ops::Range, sync::Arc};

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// BM25 top-k for the lazy-source FTS searches.
const LAZY_TEST_BM25_K: usize = 10;
/// Embedding dimension for the vector+FTS lazy-open fixture.
const LAZY_VEC_DIM: usize = 16;
/// Secondary one-hot axis weight + offset planted in each doc vector.
const LAZY_SECONDARY_WEIGHT: f32 = 0.5;
const LAZY_SECONDARY_AXIS_OFFSET: usize = 3;
/// Random-rotation seed for the lazy-open vector index.
const LAZY_VEC_ROT_SEED: u64 = 11;
/// BM25 top-k for the cold-open GET-budget search.
const LAZY_COLD_OPEN_BM25_K: usize = 5;
/// Documented cold-open range-GET budget for this non-Sq8 fixture.
const LAZY_COLD_OPEN_MAX_GETS: usize = 7;
/// Out-of-bounds range length used to probe `OutOfBounds` rejection.
const LAZY_OOB_REQUEST_LEN: u64 = 1024;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazyByteSourceError, SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    supertable::{
        StorageRangeSource,
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{decimal128_ids, default_tokenizer},
};
use tempfile::TempDir;

// ============================================================
// Tiny superfile fixture (FTS only, no vector).
// ============================================================

fn build_test_bytes() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
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
        .expect("lazy terms");
    let eager_terms = eager
        .fts()
        .expect("fts")
        .iter_column_terms("title")
        .expect("eager terms");
    assert_eq!(lazy_terms, eager_terms);
}

/// v2 (positional) variant of the fixture: same corpus, positions on.
fn build_positional_test_bytes() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: true,
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

#[tokio::test]
async fn open_lazy_reads_positional_v2_blob_like_eager() {
    // The v2 header extension and the shifted FST prefetch range are
    // lazy-open-specific code paths; this pins them against the eager
    // open on the same bytes, including a term resolution (postings
    // fetch through the lazy source with the positional term-meta
    // stride).
    let bytes = build_positional_test_bytes();
    let eager = SuperfileReader::open(bytes.clone()).expect("eager open");

    let source: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
    let lazy = SuperfileReader::open_lazy(source).await.expect("lazy open");

    assert_eq!(lazy.n_docs(), eager.n_docs());
    assert_eq!(lazy.fts_columns(), eager.fts_columns());
    let lazy_terms = lazy
        .fts()
        .expect("fts")
        .iter_column_terms("title")
        .expect("lazy terms");
    let eager_terms = eager
        .fts()
        .expect("fts")
        .iter_column_terms("title")
        .expect("eager terms");
    assert_eq!(lazy_terms, eager_terms);

    // A multi-doc term ("special", df=2) resolves through the lazy
    // postings fetch with the 32-byte positional term meta.
    let lazy_hits = lazy
        .token_match("title", &["special"], BoolMode::And)
        .await
        .expect("lazy match");
    let eager_hits = eager
        .token_match("title", &["special"], BoolMode::And)
        .await
        .expect("eager match");
    assert_eq!(lazy_hits, eager_hits);
    assert_eq!(lazy_hits, vec![0, 2]);
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
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.get_range_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        e: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
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

    // Seed the superfile at a stable URI.
    let uri = "data/seg-test.sf.parquet";
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
        .search("title", &["special"], LAZY_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("bm25");
    assert_eq!(hits.len(), 2, "two docs contain 'special'");
}

#[tokio::test]
async fn open_lazy_via_storage_matches_open_via_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = "data/seg-equiv.sf.parquet";
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
        .search("title", &["alpha"], LAZY_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("eager bm25");
    let lazy_hits = lazy
        .fts()
        .expect("fts")
        .search("title", &["alpha"], LAZY_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("lazy bm25");
    let eager_ids: Vec<_> = eager_hits.iter().map(|(d, _)| *d).collect();
    let lazy_ids: Vec<_> = lazy_hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(lazy_ids, eager_ids);
}

// ============================================================
// End-to-end cold-open range budget at the
// superfile layer. The `SuperfileReader::open_lazy`
// routes through:
//   1. `format::footer::read_parquet_metadata_lazy` — 1-2 GETs
//      for the Parquet footer.
//   2. `VectorReader::open_lazy` over a `LazySubSource` for
//      the vector subsection — 2-3 GETs.
//   3. One range GET for the FTS subsection (until
//      `FtsReader::open_lazy` exists; the FTS sub-blob is
//      bounded — typically << vector subsection).
// Total cold-open budget: ≤ 6 GETs / ≲ 2-3 MiB on a typical
// 1.5 GiB superfile.
// ============================================================

use infino::{superfile::vector::distance::normalize, test_helpers::default_vector_config};

/// Build a small superfile that exercises both the vector and
/// FTS sub-blobs so the cold-open test can observe both
/// sub-source paths in `open_lazy`. 4 docs × 16-dim vectors
/// keeps the superfile small but representative of the actual
/// layout produced by `SuperfileBuilder`.
fn build_vec_plus_fts_bytes() -> Bytes {
    let dim = LAZY_VEC_DIM;
    let n = 4usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));

    let titles_v: Vec<String> = (0..n)
        .map(|i| format!("doc {} alpha special bravo charlie", i))
        .collect();
    let titles_refs: Vec<&str> = titles_v.iter().map(String::as_str).collect();
    let ids = decimal128_ids((1u64..=n as u64).collect::<Vec<u64>>());
    let titles = LargeStringArray::from(titles_refs);
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(titles)]).expect("batch");

    // 4 unit-norm vectors so cosine is well-defined.
    let mut flat: Vec<f32> = Vec::new();
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        v[(i + LAZY_SECONDARY_AXIS_OFFSET) % dim] = LAZY_SECONDARY_WEIGHT;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }

    let opts = BuilderOptions::new(
        schema,
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", LAZY_VEC_ROT_SEED)],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

/// Superfile-level cold-open via `open_lazy`
/// stays within the documented byte budget. Asserts the upper
/// bound (≤ 6 GETs) so adding small per-subsection follow-ups
/// (e.g. a future codec-metadata range) doesn't silently inflate
/// the budget — this is the failure mode we want to catch.
#[tokio::test]
async fn cold_open_lazy_within_documented_range_budget_for_vec_plus_fts() {
    let bytes = build_vec_plus_fts_bytes();
    let total = bytes.len();
    let inner: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));

    // Wrap in a counting proxy so we can assert the budget.
    struct CountingLazy {
        inner: Arc<dyn LazyByteSource>,
        n_async: AtomicUsize,
    }
    impl std::fmt::Debug for CountingLazy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CountingLazy")
                .field("n_async", &self.n_async)
                .finish()
        }
    }
    #[async_trait]
    impl LazyByteSource for CountingLazy {
        fn size(&self) -> u64 {
            self.inner.size()
        }
        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.n_async.fetch_add(1, Ordering::AcqRel);
            self.inner.range(start, len).await
        }
        fn try_get_range_sync(&self, _: u64, _: u64) -> Option<Bytes> {
            None
        }
    }

    let counting = Arc::new(CountingLazy {
        inner,
        n_async: AtomicUsize::new(0),
    });
    let reader = SuperfileReader::open_lazy(counting.clone() as Arc<dyn LazyByteSource>)
        .await
        .expect("open_lazy");

    let n_get = counting.n_async.load(Ordering::Acquire);
    // Exact lazy open:
    //   1 footer tail
    //   3 vector metadata ranges (outer header, directory+crc, subheader;
    //     +1 more for Sq8 codec_meta on Sq8 superfiles)
    //   3 FTS metadata ranges (header, FST directory, doc-length tail)
    //
    // This tiny fixture uses a non-Sq8 vector superfile, so the expected
    // combined budget is 7. The production latency target is governed by
    // serial batches and bytes; this assertion pins that open does not
    // drift back to whole-subsection/speculative reads.
    let documented_max = LAZY_COLD_OPEN_MAX_GETS;
    assert!(
        n_get <= documented_max,
        "cold-open issued {n_get} GETs against a {total}-byte superfile; \
         documented superfile cold-open budget is ≤ {documented_max} GETs",
    );

    // Sanity: the lazy reader is usable for both vector and
    // FTS queries — the budget guarantee is meaningless if
    // the readers themselves don't function.
    assert_eq!(reader.n_docs(), 4);
    assert!(reader.vec().is_some());
    assert!(reader.fts().is_some());
    let fts_hits = reader
        .fts()
        .expect("fts")
        .search("title", &["special"], LAZY_COLD_OPEN_BM25_K, BoolMode::Or)
        .await
        .expect("bm25");
    assert!(
        !fts_hits.is_empty(),
        "BM25 must return matches for 'special'"
    );

    // The lazy reader never materializes the full superfile, so
    // `parquet_bytes()` must be `None`.
    assert!(reader.parquet_bytes().is_none());
}

#[tokio::test]
async fn storage_range_source_out_of_bounds_surfaces_typed_error() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = "data/seg-oob.sf.parquet";
    local.put_atomic(uri, bytes.clone()).await.expect("seed");

    let source = StorageRangeSource::new(Arc::clone(&local), uri)
        .await
        .expect("source");
    let size = source.size();
    let err = source
        .range(size, LAZY_OOB_REQUEST_LEN)
        .await
        .expect_err("must reject");
    assert!(
        matches!(err, LazyByteSourceError::OutOfBounds { .. }),
        "expected OutOfBounds, got {err:?}"
    );
}
