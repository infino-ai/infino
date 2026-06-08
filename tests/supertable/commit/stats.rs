// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Supertable::stats()`.
//!
//! Covers:
//!   - Fresh `create` returns the empty-supertable stats:
//!     `manifest_id == 0`, `n_superfiles == 0`,
//!     `n_manifest_parts == None`.
//!   - Stats track commits: each `writer.commit()` advances
//!     `manifest_id` + grows `n_superfiles`.
//!   - With storage attached, after the first commit
//!     `n_manifest_parts == Some(1)` and
//!     `n_manifest_parts_loaded == 0` (the writer's commit
//!     path doesn't currently hydrate the part it just wrote
//!     — `Supertable::open` is what populates the parts
//!     cache; the writer just rebuilds the in-memory state
//!     from `new_segment_list`).
//!   - `Supertable::open`'s eager-fetch populates
//!     `n_manifest_parts_loaded == n_manifest_parts`.
//!   - `process_rss_bytes` is non-zero and falls within a
//!     self-calibrating bracket around independent readings from
//!     the `memory-stats` crate taken before and after the call
//!     (i.e., the accessor is consistent).
//!   - Repeat calls return updated values (no internal
//!     caching of the snapshot).

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::supertable::Supertable;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options};

/// Disk-cache byte budget (1 GiB) for the stats integration cache.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams.
const COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch range chunk size (1 MiB).
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;
use tempfile::TempDir;

#[test]
fn fresh_supertable_returns_empty_stats() {
    let st = Supertable::create(default_supertable_options()).expect("create");
    let s = st.stats();
    assert_eq!(s.manifest_id, 0);
    assert_eq!(s.n_superfiles, 0);
    assert_eq!(
        s.n_manifest_parts, None,
        "fresh in-process supertable has no ManifestList"
    );
    assert_eq!(s.n_manifest_parts_loaded, 0);
    assert!(s.process_rss_bytes > 0, "RSS must be non-zero");
}

#[test]
fn stats_track_commits_on_in_process_supertable() {
    let st = Supertable::create(default_supertable_options()).expect("create");

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }
    let s1 = st.stats();
    assert_eq!(s1.manifest_id, 1);
    assert!(s1.n_superfiles >= 1);

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["beta"])).expect("append");
        w.commit().expect("commit");
    }
    let s2 = st.stats();
    assert_eq!(s2.manifest_id, 2);
    assert!(
        s2.n_superfiles > s1.n_superfiles,
        "commit must grow n_superfiles: {} → {}",
        s1.n_superfiles,
        s2.n_superfiles
    );
    assert_eq!(
        s2.n_manifest_parts, None,
        "in-process supertable never has a ManifestList"
    );
}

#[test]
fn stats_show_manifest_parts_when_storage_attached() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create");
    {
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["initial"])).expect("append");
        w.commit().expect("commit");
    }

    // Producer's in-memory state after commit: list is set,
    // parts cache is empty (writer rebuilds state via
    // new_segment_list, doesn't hydrate the freshly-written
    // part). Contract: report what's actually in memory.
    let producer_stats = producer.stats();
    assert_eq!(producer_stats.manifest_id, 1);
    assert_eq!(
        producer_stats.n_manifest_parts,
        Some(1),
        "post-commit ManifestList exists with one part"
    );
    assert_eq!(
        producer_stats.n_manifest_parts_loaded, 0,
        "writer's commit path doesn't hydrate the parts cache"
    );
    drop(producer);

    // Open-side: Supertable::open eager-fetches all parts, so
    // n_manifest_parts_loaded should equal n_manifest_parts.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    let consumer_stats = consumer.stats();
    assert_eq!(consumer_stats.manifest_id, 1);
    assert_eq!(consumer_stats.n_manifest_parts, Some(1));
    assert_eq!(
        consumer_stats.n_manifest_parts_loaded, 1,
        "open eager-fetches every part"
    );
}

#[test]
fn process_rss_bytes_matches_independent_reading_within_bracket() {
    // Both stats() and memory-stats::memory_stats() read the
    // same OS-reported RSS via the same crate, so back-to-back
    // calls should match closely. Under parallel cargo-test
    // execution, however, RSS is whole-process and drifts
    // because *other* tests on the same binary are allocating
    // concurrently — a fixed ±10% bound on a single independent
    // reading is not robust to that drift.
    //
    // Sandwich the stats() reading between two independent
    // reads, then bound stats() by [min(i1, i2), max(i1, i2)]
    // expanded by the natural drift between i1 and i2 plus a
    // small absolute slack for short-lived intra-syscall
    // allocations. This makes the tolerance self-calibrating to
    // whatever concurrent allocator activity the process is
    // experiencing during the test run.
    let st = Supertable::create(default_supertable_options()).expect("create");

    let read = || {
        memory_stats::memory_stats()
            .map(|m| m.physical_mem as u64)
            .expect("RSS available")
    };

    let i1 = read();
    let s = st.stats().process_rss_bytes;
    let i2 = read();

    assert!(s > 0, "stats.process_rss_bytes must be non-zero");
    assert!(
        i1 > 0 && i2 > 0,
        "independent RSS readings must be non-zero"
    );

    // Slack = the drift observed between i1 and i2 (concurrent
    // process activity) + 64 MiB absolute floor for in-between
    // allocations the test thread itself may incur.
    const ABS_SLACK_BYTES: u64 = 64 * 1024 * 1024;
    let drift = i1.abs_diff(i2);
    let slack = drift.saturating_add(ABS_SLACK_BYTES);
    let lo = i1.min(i2).saturating_sub(slack);
    let hi = i1.max(i2).saturating_add(slack);
    assert!(
        s >= lo && s <= hi,
        "stats.process_rss_bytes={s} outside [{lo}, {hi}] \
         (independent reads: {i1}, {i2}; drift={drift}, slack={slack})"
    );
}

#[test]
fn repeat_stats_calls_return_fresh_snapshots() {
    // No internal caching: calling stats() twice after a
    // mutation must reflect the mutation.
    let st = Supertable::create(default_supertable_options()).expect("create");
    let pre = st.stats();
    assert_eq!(pre.manifest_id, 0);

    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["something"]))
            .expect("append");
        w.commit().expect("commit");
    }
    let post = st.stats();
    assert_eq!(post.manifest_id, 1);
    assert!(post.n_superfiles > pre.n_superfiles);
}

#[test]
fn stats_without_disk_cache_have_none_cache_counters() {
    // No cache attached → all cache-counter fields are None.
    // Distinguishing None vs Some(0) is the contract: a
    // consumer can tell whether "no cold fetches happened"
    // is because there's no cache at all, or because the
    // cache is attached but the workload didn't trigger one.
    let st = Supertable::create(default_supertable_options()).expect("create");
    let s = st.stats();
    assert_eq!(s.n_cold_fetches, None);
    assert_eq!(s.n_cache_evictions, None);
    assert_eq!(s.n_cache_madvise_calls, None);
    assert_eq!(s.n_cache_entries, None);
    assert_eq!(s.mmap_resident_bytes, None);
    assert_eq!(s.memory_budget_bytes, None);
}

#[test]
fn stats_with_disk_cache_attached_surface_zero_counters_on_fresh_cache() {
    // Cache attached, nothing read through it yet → all
    // four counter fields are Some(0). This is the
    // contract: cold-fetch / eviction / madvise / entry
    // counts surface through `Supertable::stats()` even
    // before any activity, so downstream consumers can
    // sample them on a timer without worrying about
    // initialization order.
    use infino::supertable::SuperfileUri;
    use infino::supertable::reader_cache::{
        ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy,
    };
    use std::collections::HashSet;

    let storage_dir = TempDir::new().expect("storage dir");
    let cache_dir = TempDir::new().expect("cache dir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync> = Arc::new(HashSet::new);
    let cache = DiskCacheStore::new(Arc::clone(&storage), cfg, pinned).expect("cache");

    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_disk_cache(Arc::clone(&cache));
    let st = Supertable::create(opts).expect("create");

    let s = st.stats();
    assert_eq!(s.n_cold_fetches, Some(0), "fresh cache: zero cold fetches");
    assert_eq!(s.n_cache_evictions, Some(0));
    assert_eq!(s.n_cache_madvise_calls, Some(0));
    assert_eq!(s.n_cache_entries, Some(0));
    // mmap_resident_bytes is also Some when cache attached.
    assert!(
        s.mmap_resident_bytes.is_some(),
        "mmap_resident_bytes surfaces when cache is attached"
    );
}
