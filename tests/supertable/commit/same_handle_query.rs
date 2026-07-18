// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Same-handle commit → query against a storage-backed table.
//!
//! Regression coverage for the commit-path manifest the writer
//! installs in `inner.manifest` after a commit (via
//! `ManifestSnapshot::rebalance`). The producer handle keeps querying its
//! own committed state without reopening, so the post-commit
//! manifest must be able to resolve the parts it just wrote:
//!
//! - It carries a loader (built from the new list against
//!   `options.storage`), so a freshly *created* table — whose
//!   initial manifest has no loader — can still load parts after
//!   its first commit. Previously this surfaced as
//!   `ManifestLoadError::NoLoaderAttached`.
//! - It seeds the freshly-written parts into the in-memory cache,
//!   so the first same-handle query reads zero manifest parts back
//!   from storage.
//! - A second commit's rebalance can load + rewrite the prior
//!   part (the rewrite path calls `get_part_by_id` on the
//!   in-memory manifest), so commit-after-commit doesn't fault.
//!
//! `Consistency::Snapshot` keeps the producer from re-checking the
//! pointer, so every read here is served by the exact manifest the
//! commit installed — not a refreshed/reopened one that would mask
//! a loaderless commit manifest.

#![deny(clippy::unwrap_used)]

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        options::Consistency,
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

/// BM25 top-k for the regression queries.
const BM25_TOP_K: usize = 10;
/// URI prefix manifest-part objects live under; used to isolate
/// part GETs from superfile-data GETs in the refetch assertion.
const MANIFEST_PARTS_PREFIX: &str = "manifests/";

#[test]
fn query_after_first_commit_on_same_handle_succeeds() {
    // A freshly *created* table's initial manifest has no loader.
    // The first commit must install one, or this query fails with
    // NoLoaderAttached.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_read_consistency(Consistency::Snapshot),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    let hits = st
        .reader()
        .bm25_hits("title", "alpha", BM25_TOP_K, BoolMode::Or)
        .expect("same-handle query after first commit must resolve parts");
    assert_eq!(hits.len(), 1, "expected the one matching row");
}

#[test]
fn query_after_second_commit_on_same_handle_succeeds() {
    // The second commit's rebalance loads + rewrites the prior
    // part (rewrite path), then installs a manifest that must
    // resolve both the carried-over and newly-written rows.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_read_consistency(Consistency::Snapshot),
    )
    .expect("create");

    let mut w = st.writer().expect("writer 1");
    w.append(&build_title_batch(&["alpha bravo"]))
        .expect("append 1");
    w.commit().expect("commit 1");
    drop(w);

    let mut w = st.writer().expect("writer 2");
    w.append(&build_title_batch(&["echo foxtrot"]))
        .expect("append 2");
    w.commit().expect("commit 2");
    drop(w);

    // A term from the first commit (survives the part rewrite)...
    let old_hits = st
        .reader()
        .bm25_hits("title", "alpha", BM25_TOP_K, BoolMode::Or)
        .expect("query for first-commit term");
    assert_eq!(
        old_hits.len(),
        1,
        "first-commit row must survive the rewrite"
    );

    // ...and a term from the second commit.
    let new_hits = st
        .reader()
        .bm25_hits("title", "echo", BM25_TOP_K, BoolMode::Or)
        .expect("query for second-commit term");
    assert_eq!(new_hits.len(), 1, "second-commit row must be queryable");
}

#[test]
fn same_handle_query_after_commit_refetches_no_manifest_parts() {
    // The commit path seeds the freshly-written parts into the
    // in-memory cache, so the first same-handle query issues zero
    // GETs against the manifest-parts namespace.
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let counter = Arc::new(PartGetCounter::new(local));
    let storage: Arc<dyn StorageProvider> = Arc::clone(&counter) as Arc<dyn StorageProvider>;
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(storage)
            .with_read_consistency(Consistency::Snapshot),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    let before = counter.part_gets();
    let hits = st
        .reader()
        .bm25_hits("title", "alpha", BM25_TOP_K, BoolMode::Or)
        .expect("query");
    let after = counter.part_gets();

    assert_eq!(hits.len(), 1);
    assert_eq!(
        after - before,
        0,
        "post-commit query must not refetch manifest parts (seeded into cache)"
    );
}

#[test]
fn parts_cache_stays_bounded_across_repeated_commits() {
    // The commit-path manifest inherits only the parts the new list
    // still references (plus the freshly-written ones), so repeated
    // commits that rewrite the same partition don't accumulate the
    // superseded part versions in the in-memory cache. Before the
    // fix, `n_manifest_parts_loaded` grew by one per commit while
    // `n_manifest_parts` stayed at 1.
    const COMMITS: usize = 6;
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(storage)
            .with_read_consistency(Consistency::Snapshot),
    )
    .expect("create");

    for i in 0..COMMITS {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha bravo"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w);

        let s = st.stats();
        // The cache never holds more parts than the live list — no
        // superseded part versions linger across commits.
        assert!(
            s.n_manifest_parts_loaded <= s.n_manifest_parts,
            "commit {}: cache ({}) exceeds live parts ({}) — superseded \
             parts are accumulating",
            i + 1,
            s.n_manifest_parts_loaded,
            s.n_manifest_parts,
        );
    }
}

// ============================================================
// Metadata-GET law: open costs a fixed pointer + list + part
// fetch. On the read path the ONLY metadata I/O is the
// BoundedStaleness freshness check — one pointer GET at most
// once per window, which short-circuits (`AlreadyLoaded`)
// before any list/part fetch when the pointer hasn't advanced.
// Lists and parts are never re-fetched by reads.
// ============================================================

/// Staleness window wide enough that only the FIRST query's freshness
/// check can fire during the test — every later read must be GET-free.
const WIDE_STALENESS_SECS: u64 = 3600;

#[test]
fn open_metadata_cost_is_fixed_and_reads_check_pointer_once_per_window() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer: two commits so multiple parts/entries exist.
    {
        let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&local)))
            .expect("create");
        for text in ["alpha bravo", "charlie delta"] {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&[text])).expect("append");
            w.commit().expect("commit");
        }
    }

    // Consumer behind the counting proxy.
    let counter = Arc::new(PartGetCounter::new(local));
    let storage: Arc<dyn StorageProvider> = Arc::clone(&counter) as Arc<dyn StorageProvider>;
    let st = Supertable::open(
        default_supertable_options()
            .with_storage(storage)
            .with_read_consistency(Consistency::BoundedStaleness(
                std::time::Duration::from_secs(WIDE_STALENESS_SECS),
            )),
    )
    .expect("open");

    // Open cost is fixed and small: one pointer GET, one list GET, and the
    // eager part fetch.
    assert_eq!(
        counter.pointer_gets(),
        1,
        "open must read the pointer object exactly once"
    );
    assert_eq!(counter.list_gets(), 1, "open fetches the list by URI once");
    assert_eq!(
        counter.part_gets(),
        1,
        "single-bucket table: open eager-loads exactly one part"
    );

    // Reads: the FIRST query pays the one freshness check for the
    // staleness window — a single pointer GET whose unchanged pointer
    // short-circuits (`AlreadyLoaded`) before any list/part fetch.
    // Every further read inside the window issues zero metadata GETs;
    // the resident flat view serves them all.
    for _ in 0..3 {
        let hits = st
            .reader()
            .bm25_hits("title", "alpha", BM25_TOP_K, BoolMode::Or)
            .expect("query");
        assert_eq!(hits.len(), 1);
    }
    assert_eq!(
        counter.pointer_gets(),
        2,
        "reads perform exactly one freshness pointer check per staleness window"
    );
    assert_eq!(counter.list_gets(), 1, "reads must never fetch a list");
    assert_eq!(counter.part_gets(), 1, "reads must never fetch a part");
}

/// Storage proxy that counts `get`s into the manifest metadata
/// namespaces (pointer / lists / parts), delegating everything else
/// to the inner provider.
#[derive(Debug)]
struct PartGetCounter {
    inner: Arc<dyn StorageProvider>,
    part_gets: AtomicUsize,
    list_gets: AtomicUsize,
    pointer_gets: AtomicUsize,
}

impl PartGetCounter {
    fn new(inner: Arc<dyn StorageProvider>) -> Self {
        Self {
            inner,
            part_gets: AtomicUsize::new(0),
            list_gets: AtomicUsize::new(0),
            pointer_gets: AtomicUsize::new(0),
        }
    }

    fn part_gets(&self) -> usize {
        self.part_gets.load(Ordering::Acquire)
    }

    fn list_gets(&self) -> usize {
        self.list_gets.load(Ordering::Acquire)
    }

    fn pointer_gets(&self) -> usize {
        self.pointer_gets.load(Ordering::Acquire)
    }
}

#[async_trait]
impl StorageProvider for PartGetCounter {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        if uri.starts_with(MANIFEST_PARTS_PREFIX) || uri.starts_with("manifest-parts/") {
            self.part_gets.fetch_add(1, Ordering::AcqRel);
        }
        if uri.starts_with("manifest/") {
            self.list_gets.fetch_add(1, Ordering::AcqRel);
        }
        if uri == "_supertable/current" {
            self.pointer_gets.fetch_add(1, Ordering::AcqRel);
        }
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.inner.get_range(uri, range).await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, expected).await
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
