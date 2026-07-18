// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-cache integration in the supertable
//! reader path.
//!
//! Covers the load-bearing invariants:
//!
//!   - **Cache routing on cross-process consumer.** A
//!     producer commits superfiles through storage + cache.
//!     The producer drops; a fresh `Supertable::open`
//!     consumer (with its own cache, sharing the storage
//!     root) queries via SQL → reader bytes flow through
//!     `DiskCacheStore::reader` → cold-fetch from storage.
//!     The cache's `n_cold_fetches` counter grows.
//!   - **Warm-hit on second query.** A repeat of the same
//!     query against the consumer hits the warm cache —
//!     `n_cold_fetches` is unchanged.
//!   - **Skip-in-memory writer path doesn't break local
//!     reads.** The producer queries its own superfiles
//!     after committing. With cache attached, the writer
//!     bypasses `options.store.put`; the cache hydrates
//!     lazily on the producer's first read.
//!   - **No-cache path unchanged.** A supertable without
//!     `with_disk_cache` still uses the in-memory store
//!     directly (no cache plumbing engaged). This is the
//!     legacy path; the existing test suite continues to
//!     cover it. We assert it here too as a regression
//!     check.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::RecordBatch;
use infino::{
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_disk_cache, default_supertable_options},
};
use tempfile::TempDir;

/// Commits driven for the warm-cache idempotency test.
const WARM_CACHE_COMMIT_COUNT: usize = 3;
/// 1-byte memory budget forcing the post-commit madvise sweep.
const MEMORY_BUDGET_FORCE_SWEEP_BYTES: u64 = 1;

#[test]
fn cross_process_consumer_routes_reads_through_disk_cache() {
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));

    // ---- Producer: commit + drop. -----------------------------
    // Producer doesn't need a cache for this test — we're
    // validating the consumer's read path.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("producer writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("producer commit");
    }

    // ---- Consumer: open with disk cache attached. -------------
    let consumer_cache_dir = TempDir::new().expect("cache tempdir");
    let cache = default_disk_cache(Arc::clone(&storage), consumer_cache_dir.path());
    let consumer_opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_disk_cache(Arc::clone(&cache));
    let consumer = Supertable::open(consumer_opts).expect("open");
    assert_eq!(consumer.manifest_id(), 1);

    // Cache starts cold: zero cold-fetches.
    let pre_stats = cache.stats();
    assert_eq!(pre_stats.n_cold_fetches, 0);
    assert_eq!(pre_stats.n_entries, 0);

    // First query against the consumer routes through the
    // disk cache → cold-fetch from storage.
    let batches = consumer
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY _id")
        .expect("first query");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);

    let mid_stats = cache.stats();
    assert!(
        mid_stats.n_cold_fetches >= 1,
        "first query must cold-fetch through the cache; got n_cold_fetches={}",
        mid_stats.n_cold_fetches
    );
    assert!(
        mid_stats.n_entries >= 1,
        "cache must hold at least one entry after cold-fetch; got n_entries={}",
        mid_stats.n_entries
    );
    assert!(
        mid_stats.current_bytes > 0,
        "cache must hold some bytes after cold-fetch; got current_bytes={}",
        mid_stats.current_bytes
    );

    // Second query against the same superfile — warm hit.
    let _batches = consumer
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY _id")
        .expect("second query");
    let post_stats = cache.stats();
    assert_eq!(
        post_stats.n_cold_fetches, mid_stats.n_cold_fetches,
        "second query must hit the warm cache; cold-fetches grew from {} to {}",
        mid_stats.n_cold_fetches, post_stats.n_cold_fetches
    );
}

#[test]
fn producer_with_cache_reads_through_cache_path() {
    // The producer commits with both storage AND cache
    // attached. The writer extracts
    // summaries directly from the superfile bytes (no
    // round-trip through options.store.put), so the
    // in-memory tier is NOT populated. The
    // writer additionally pre-populates the cache after
    // commit succeeds → the producer's own first query
    // hits the warm cache.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    let producer = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");
    let mut w = producer.writer().expect("writer");
    w.append(&build_title_batch(&["alpha"])).expect("append");
    w.commit().expect("commit");
    drop(w);

    // Post-commit the cache holds the warmed
    // superfile. n_cold_fetches stays 0; n_entries == 1.
    let pre = cache.stats();
    assert_eq!(pre.n_cold_fetches, 0);
    assert_eq!(pre.n_entries, 1, "writer should have warmed the cache");

    let batches = producer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query");
    assert_eq!(batches.len(), 1);
    let post = cache.stats();
    assert_eq!(
        post.n_cold_fetches, 0,
        "producer's query must hit the warm cache; got n_cold_fetches={}",
        post.n_cold_fetches
    );
}

#[test]
fn writer_warms_cache_on_commit_so_producer_query_skips_cold_fetch() {
    // With cache attached, the writer pre-populates
    // the cache after each successful commit. The
    // producer's own queries on its just-committed superfiles
    // hit the warm cache directly — no cold-fetch
    // round-trip through storage.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");

    // Pre-commit: cache empty.
    assert_eq!(cache.stats().n_entries, 0);
    assert_eq!(cache.stats().n_cold_fetches, 0);

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }

    // Post-commit: writer pre-populated the cache.
    let post_commit = cache.stats();
    assert_eq!(
        post_commit.n_entries, 1,
        "writer must have warmed the cache with the just-committed superfile"
    );
    assert_eq!(
        post_commit.n_cold_fetches, 0,
        "warming is a direct insert; no cold-fetches recorded"
    );
    assert!(
        post_commit.current_bytes > 0,
        "cache must hold the warmed superfile's bytes; got current_bytes={}",
        post_commit.current_bytes
    );

    // Producer's query hits the warm cache — no cold-fetch.
    let batches = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query");
    assert_eq!(batches.len(), 1);

    let post_query = cache.stats();
    assert_eq!(
        post_query.n_cold_fetches, 0,
        "producer query must hit warm cache; cold-fetches stayed at 0"
    );
    assert_eq!(post_query.n_entries, 1);
}

#[test]
fn writer_warm_cache_is_idempotent_under_writer_retry() {
    // The writer's warm-insert path swallows
    // already-cached URIs as a no-op. Sanity-check by
    // committing twice with the same supertable handle —
    // the second commit's pre-populated entries are
    // distinct URIs (UUID v4), so each lands a fresh entry,
    // but the cache itself never panics or double-inserts.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");

    for _i in 0..WARM_CACHE_COMMIT_COUNT {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["title"])).expect("append");
        w.commit().expect("commit");
    }

    // 3 commits → 3 warm-inserted entries → 0 cold-fetches.
    let stats = cache.stats();
    assert_eq!(stats.n_entries, 3);
    assert_eq!(stats.n_cold_fetches, 0);
}

#[test]
fn manifest_superfiles_are_not_pinned_by_supertable_create() {
    // The cache must be free to evict any superfile to stay under
    // budget. Pinning the live manifest makes the whole index
    // required to fit in cache: once all resident entries are pinned,
    // the next admit can fail with BudgetExceeded. Supertable::create
    // therefore installs a pin callback that does not pin manifest
    // superfiles.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");

    // Before any commit: nothing pinned.
    let pre = cache.current_pinned_uris();
    assert!(pre.is_empty(), "expected empty pinned set; got {pre:?}");

    // Commit one superfile.
    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["pinned alpha"]))
            .expect("append");
        w.commit().expect("commit");
    }

    // Post-commit: the manifest has one superfile, but the cache still
    // pins nothing. Eviction protection is not based on the live
    // manifest set.
    let post = cache.current_pinned_uris();
    assert!(post.is_empty(), "expected empty pinned set; got {post:?}");
    let reader = st.reader();
    assert_eq!(reader.manifest().get_all_superfiles().len(), 1);
}

#[test]
fn manifest_superfiles_are_not_pinned_by_supertable_open() {
    // Supertable::open follows the same cache policy as create: the
    // live manifest does not pin superfile files. A cross-process
    // consumer can stream an index larger than cache budget instead
    // of making every manifest superfile ineligible for eviction.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));

    // Producer (no cache attached): commit + drop.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["from producer"]))
            .expect("append");
        w.commit().expect("commit");
    }

    // Consumer with cache.
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    let pinned = cache.current_pinned_uris();
    assert!(
        pinned.is_empty(),
        "expected empty pinned set; got {pinned:?}"
    );
    let n_superfiles = consumer.reader().manifest().get_all_superfiles().len();
    assert_eq!(n_superfiles, 1);
}

#[test]
fn pinned_fn_does_not_hold_supertable_alive() {
    // Cache outliving its supertable is supported: the installed
    // callback must not keep the supertable alive. The current policy
    // pins no manifest superfiles, so the observable set is empty both
    // before and after drop.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    {
        let st = Supertable::create(
            default_supertable_options()
                .with_storage(Arc::clone(&storage))
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["temp"])).expect("append");
        w.commit().expect("commit");
        assert!(cache.current_pinned_uris().is_empty());
        // st drops at end of scope.
    }

    // Supertable dropped; cache survives without pinning anything.
    assert!(
        cache.current_pinned_uris().is_empty(),
        "pinned set must be empty after supertable drops"
    );
}

#[test]
fn memory_budget_drives_post_commit_madvise_sweep() {
    // With both with_disk_cache + with_memory_budget,
    // the writer's post-commit path triggers
    // sweep_for_budget. With a budget below the working
    // set, the n_madvise_calls counter grows; without a
    // budget, it doesn't.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    // Budget of 1 byte → every commit's mmap exceeds it
    // → sweep fires.
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache))
            .with_memory_budget(MEMORY_BUDGET_FORCE_SWEEP_BYTES),
    )
    .expect("create");

    let before = cache.stats().n_madvise_calls;

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }

    let after = cache.stats().n_madvise_calls;
    assert!(
        after > before,
        "post-commit sweep must have called madvise; n_madvise_calls {} → {}",
        before,
        after
    );

    // Confirm the budget + mmap_resident shows up in stats.
    let stats = st.stats();
    assert_eq!(stats.memory_budget_bytes, Some(1));
    assert!(stats.mmap_resident_bytes.is_some());
}

#[test]
fn memory_budget_unset_does_not_force_sweep() {
    // Without with_memory_budget, the post-commit sweep
    // doesn't fire — the cache's idle-threshold sweep
    // is the only mechanism, and a sub-second test won't
    // trigger it.
    let storage_dir = TempDir::new().expect("storage tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");

    let before = cache.stats().n_madvise_calls;

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }

    let after = cache.stats().n_madvise_calls;
    assert_eq!(
        after, before,
        "no budget → no forced sweep; n_madvise_calls {} → {}",
        before, after
    );

    let stats = st.stats();
    assert_eq!(stats.memory_budget_bytes, None);
}

#[test]
fn no_cache_path_still_uses_in_memory_store() {
    // Regression check: a supertable WITHOUT
    // with_disk_cache still queries via the in-memory
    // store. This is the in-memory / legacy path; covered by the
    // broader test suite, but reasserted here so a future
    // refactor that accidentally engages the cache plumbing
    // unconditionally is caught.
    let st = Supertable::create(default_supertable_options()).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["only"])).expect("append");
    w.commit().expect("commit");

    let batches = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    // The cache field is None; the path can't engage it.
    // No counters to assert; the legacy in-memory store
    // doesn't expose any.
}
