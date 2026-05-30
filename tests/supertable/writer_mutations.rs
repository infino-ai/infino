//! `SupertableWriter::delete` integration tests.
//!
//! Drive the public mutation API end-to-end: append rows, delete
//! by predicate, verify subsequent queries don't see the deleted
//! rows. The buffered + commit shape from the plan is a follow-up;
//! this commit ships immediate-drive deletes.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::Array;
use datafusion::prelude::{Expr, col, lit};
use tempfile::TempDir;

use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::superfile::fts::reader::BoolMode;
use infino::supertable::Supertable;
use infino::supertable::mutations::MutationError;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::test_helpers::{build_title_batch, default_supertable_options};

fn make_disk_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: 1 << 30,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: 4,
        cold_fetch_chunk_bytes: 1 << 20,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
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
    );

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "alpha",
        "bravo",
        "charlie",
        "alpha delta",
    ]))
    .expect("append");
    w.commit().expect("commit");

    // Delete every row whose `title = 'bravo'`. Should tombstone
    // exactly 1 row.
    let predicate: Expr = col("title").eq(lit("bravo"));
    let outcome = w.delete(predicate).expect("delete");
    assert_eq!(outcome.matched, 1);
    assert_eq!(outcome.n_tombstoned, 1);
    assert_eq!(outcome.n_not_found, 0);
    drop(w);

    // Follow-up SQL query no longer returns the row.
    let batches = st
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
    // hits.
    let r = st.reader();
    let hits = r
        .bm25_search("title", "bravo", 10, BoolMode::Or)
        .expect("fts");
    assert!(hits.is_empty(), "expected zero hits for tombstoned token");
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
    );

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["x", "y"])).expect("append");
    w.commit().expect("commit");

    let outcome = w
        .delete(col("title").eq(lit("not-present")))
        .expect("delete");
    assert_eq!(outcome.matched, 0);
    assert_eq!(outcome.n_tombstoned, 0);
    assert_eq!(outcome.n_not_found, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writer_delete_requires_storage() {
    // In-memory-only supertable can't be mutated through the WAL
    // pipeline.
    let st = Supertable::create(default_supertable_options());
    let mut w = st.writer().expect("writer");
    let err = w
        .delete(col("title").eq(lit("foo")))
        .expect_err("must error");
    assert!(matches!(err, MutationError::NoStorageAttached));
}
