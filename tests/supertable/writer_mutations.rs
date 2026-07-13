// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableWriter::update` + `delete` integration tests.
//!
//! Drive the public mutation API end-to-end: buffer mutations
//! via `update` / `delete`, flush via `commit`, verify that
//! subsequent SQL + FTS queries reflect the mutation (deleted
//! rows are gone, updated rows show the replacement payload).

use std::{collections::HashSet, sync::Arc};

use arrow_array::Array;
use datafusion::prelude::{Expr, col, lit};
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        mutations::MutationError,
        options::Consistency,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

/// Disk-cache byte budget (1 GiB) for the mutation integration cache.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams for the test disk cache.
const COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch range chunk size (1 MiB).
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;
/// Background prefetch concurrency for the hybrid cache.
const PREFETCH_CONCURRENCY: usize = 8;
/// Mmap promotion timers disabled in tests (no idle eviction).
const MMAP_TIMER_DISABLED_SECS: u64 = 0;
/// BM25 top-k for post-mutation FTS queries.
const FTS_TOP_K: usize = 10;

fn make_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        prefetch_concurrency: PREFETCH_CONCURRENCY,
        mmap_cold_threshold_secs: MMAP_TIMER_DISABLED_SECS,
        mmap_sweep_interval_secs: MMAP_TIMER_DISABLED_SECS,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_delete_tombstones_matching_rows() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "alpha",
        "bravo",
        "charlie",
        "alpha delta",
    ]))
    .expect("append");
    w.commit().expect("commit");

    // Buffer a delete + commit it. PendingDelete carries the
    // call-time match count; the commit's outcome reflects how
    // many tombstones actually landed.
    let predicate: Expr = col("title").eq(lit("bravo"));
    let pending = w.delete(predicate).expect("delete");
    assert_eq!(pending.matched, 1);
    let result = w.commit().expect("commit delete");
    assert_eq!(result.outcomes.len(), 1);
    let outcome = &result.outcomes[0];
    assert_eq!(outcome.matched(), 1);
    assert_eq!(outcome.n_tombstoned(), 1);
    assert_eq!(outcome.n_not_found(), 0);
    drop(w);

    // Follow-up SQL query no longer returns the row.
    let batches = st
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY title")
        .expect("sql");
    let titles: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .expect("title col");
            (0..col.len()).map(move |i| col.value(i).to_string())
        })
        .collect();
    assert_eq!(
        titles,
        vec!["alpha".to_string(), "alpha delta".into(), "charlie".into()]
    );

    // Follow-up FTS query against the deleted token returns no
    // hits. The row-returning search yields one (possibly empty)
    // batch, so assert on the row count, not the batch count.
    let hits = st
        .reader()
        .bm25_search("title", "bravo", FTS_TOP_K, BoolMode::Or, None)
        .expect("fts");
    let n_rows: usize = hits.iter().map(|b| b.num_rows()).sum();
    assert_eq!(n_rows, 0, "expected zero hits for tombstoned token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_delete_on_predicate_with_no_matches_returns_zero_outcome() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["x", "y"])).expect("append");
    w.commit().expect("commit");

    let pending = w
        .delete(col("title").eq(lit("not-present")))
        .expect("delete");
    assert_eq!(pending.matched, 0);
    // Even a zero-match delete buffers + commits a WAL — the
    // tombstone phase has nothing to do but the WAL still
    // transitions to Complete cleanly.
    let result = w.commit().expect("commit zero-match");
    assert_eq!(result.outcomes.len(), 1);
    let outcome = &result.outcomes[0];
    assert_eq!(outcome.matched(), 0);
    assert_eq!(outcome.n_tombstoned(), 0);
    assert_eq!(outcome.n_not_found(), 0);
}

/// The cross-worker delete-propagation contract: a second handle on
/// the same storage (modeling another worker process with its own
/// tombstone cache) sees a delete on the *very next query* under
/// `Consistency::Strong` — no TTL window. The delete pipeline stamps
/// the touched superfile's tombstone seq into the manifest, the
/// reader's per-query pointer refresh picks the stamp up, and the
/// sidecar cache refetches exactly the named sidecar.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_is_visible_to_other_handles_on_next_query() {
    let dir = TempDir::new().expect("tempdir");
    let storage_a: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let storage_b: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let writer_handle =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage_a)))
            .expect("create");
    let mut w = writer_handle.writer().expect("writer");
    w.append(&build_title_batch(&["alpha", "bravo", "charlie"]))
        .expect("append");
    w.commit().expect("commit");

    // "Another worker": a separate handle with its own (empty)
    // tombstone cache, reading at strong consistency.
    let reader_handle = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage_b))
            .with_read_consistency(Consistency::Strong),
    )
    .expect("open");

    // Warm the other worker's caches with a pre-delete query so the
    // later assertion exercises invalidation, not a cold read.
    let n_rows: usize = reader_handle
        .reader()
        .bm25_search("title", "bravo", FTS_TOP_K, BoolMode::Or, None)
        .expect("pre-delete fts")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(n_rows, 1, "row visible before the delete");

    // Worker A deletes the row. Commit = sidecar CAS-PUT + manifest
    // seq stamp; the stamp is its own manifest version.
    let manifest_id_before = writer_handle.manifest_id();
    w.delete(col("title").eq(lit("bravo"))).expect("delete");
    w.commit().expect("commit delete");
    drop(w);
    assert!(
        writer_handle.manifest_id() > manifest_id_before,
        "the delete's tombstone-seq stamp publishes a new manifest version"
    );

    // The very next query on the other worker must drop the row.
    let n_rows: usize = reader_handle
        .reader()
        .bm25_search("title", "bravo", FTS_TOP_K, BoolMode::Or, None)
        .expect("post-delete fts")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(n_rows, 0, "delete visible to the other handle immediately");

    let batches = reader_handle
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY title")
        .expect("post-delete sql");
    let titles: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .expect("title col");
            (0..col.len()).map(move |i| col.value(i).to_string())
        })
        .collect();
    assert_eq!(titles, vec!["alpha".to_string(), "charlie".into()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_delete_requires_storage() {
    // In-memory-only supertable can't be mutated through the WAL
    // pipeline.
    let st = Supertable::create(default_supertable_options()).expect("create");
    let mut w = st.writer().expect("writer");
    let err = w
        .delete(col("title").eq(lit("foo")))
        .expect_err("must error");
    assert!(matches!(err, MutationError::NoStorageAttached));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_update_replaces_matching_rows() {
    // Insert 3 rows, then update the row whose title is "bravo"
    // to "bravo-prime". Post-update: 3 rows total visible; "bravo"
    // is gone; "bravo-prime" is present.
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha", "bravo", "charlie"]))
        .expect("append");
    w.commit().expect("commit");

    let new_rows = build_title_batch(&["bravo-prime"]);
    let pending = w
        .update(col("title").eq(lit("bravo")), new_rows)
        .expect("update");
    assert_eq!(pending.matched, 1);
    // Drive the buffered update through the WAL pipeline.
    let result = w.commit().expect("commit update");
    assert_eq!(result.outcomes.len(), 1);
    let outcome = &result.outcomes[0];
    assert_eq!(outcome.matched(), 1);
    assert_eq!(outcome.n_tombstoned(), 1);
    assert_eq!(outcome.n_not_found(), 0);
    drop(w);

    let batches = st
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY title")
        .expect("sql");
    let titles: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .expect("title col");
            (0..col.len()).map(move |i| col.value(i).to_string())
        })
        .collect();
    assert_eq!(
        titles,
        vec!["alpha".to_string(), "bravo-prime".into(), "charlie".into(),]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_update_cardinality_mismatch_is_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    // Insert 3 rows.
    w.append(&build_title_batch(&["a", "b", "c"]))
        .expect("append");
    w.commit().expect("commit");

    // Predicate matches 1 row; provide 2 new rows → mismatch.
    let new_rows = build_title_batch(&["one", "two"]);
    let err = w
        .update(col("title").eq(lit("a")), new_rows)
        .expect_err("must mismatch");
    assert!(matches!(
        err,
        MutationError::CardinalityMismatch {
            matched: 1,
            new_rows: 2
        }
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_emitted_superfile_carries_subsection_offsets() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let disk_cache = make_disk_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha", "bravo", "charlie"]))
        .expect("append");
    w.commit().expect("commit");
    w.update(
        col("title").eq(lit("bravo")),
        build_title_batch(&["bravo-prime"]),
    )
    .expect("update");
    w.commit().expect("commit update");
    drop(w);

    let reader = st.reader();
    let manifest = reader.manifest();
    let emitted = manifest
        .get_all_superfiles()
        .iter()
        .find(|e| e.n_docs == 1)
        .expect("update-emitted single-row superfile present");
    let offsets = emitted
        .subsection_offsets
        .as_ref()
        .expect("update-emitted superfile carries subsection_offsets");
    assert!(offsets.total_size > 0, "total_size is stamped");
}
