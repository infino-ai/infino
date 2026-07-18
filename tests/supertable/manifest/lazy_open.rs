// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Lazy part-load machinery behind the explicit
//! `with_eager_load_threshold` opt-in.
//!
//! The DEFAULT is always-eager (open fetches every part in parallel so
//! search pays no serial manifest GETs); lazy mode remains reachable by
//! setting the threshold below the part count. Covers the load-bearing
//! invariants:
//!
//!   - **Default open is eager.** The manifest's flat
//!     `superfile_list.superfiles` is populated after open, and the parts
//!     cache has the `OnceCell`s initialized.
//!   - **Forced-lazy open loads nothing.** With the threshold below the
//!     part count, open populates empty `OnceCell`s only — no part bytes
//!     fetched. `superfile_list.superfiles` stays empty.
//!   - **First `ManifestSnapshot::part(id).await` lazy-loads
//!     one.** Single storage GET for that part; the
//!     OnceCell stays populated for subsequent calls (no
//!     re-fetch on the second call).
//!   - **`with_eager_load_threshold(0)` forces lazy mode**
//!     even on a 1-part manifest — test-friendly knob.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::{
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};

/// One superfile per manifest part (forces a multi-part list).
const TARGET_SUPERFILES_PER_PARTITION: u64 = 1;
/// Multi-part shape for the forced-lazy tests. The always-eager default
/// loads any part count at open, so lazy mode is entered via
/// `with_eager_load_threshold` below this count.
const LAZY_MODE_PART_COUNT: usize = 5;
/// Threshold below [`LAZY_MODE_PART_COUNT`] that forces the multi-part
/// fixtures into lazy mode.
const EAGER_LOAD_THRESHOLD_BELOW_PARTS: u32 = 4;
/// Which 0-based part to lazy-load in the targeted-load test.
const LAZY_LOAD_TARGET_PART_INDEX: usize = 2;
/// Eager-load threshold of 0 forces lazy mode on a 1-part manifest.
const EAGER_LOAD_THRESHOLD_FORCE_LAZY: u32 = 0;
use tempfile::TempDir;

#[test]
fn one_part_eager_fetches_under_default_threshold() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer: 1 commit → 1 part.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }

    // Consumer with default threshold (4) opens a 1-part
    // manifest → eager-fetch.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");

    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(list_entries.len(), 1);
    assert_eq!(
        m.get_all_superfiles().len(),
        1,
        "eager mode must populate superfile_list.superfiles"
    );
    // Eager-mode populates the OnceCell.
    let part = m.get_cached_part_by_list_idx(0);
    assert!(
        part.is_some(),
        "eager-fetched OnceCell should be initialized"
    );
}

#[test]
fn many_parts_skip_eager_fetch() {
    // target_superfiles_per_partition=1 + 5 single-superfile
    // commits → 5 list entries, all sharing the same
    // partition_key (the partition-split path). The default is
    // always-eager, so lazy mode is forced with a threshold below
    // the part count.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let producer_opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_target_superfiles_per_part(TARGET_SUPERFILES_PER_PARTITION);
    let producer = Supertable::create(producer_opts).expect("create");
    for _i in 0..LAZY_MODE_PART_COUNT {
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
    }
    drop(producer);

    // Consumer forced lazy: threshold 4 < 5 parts.
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_BELOW_PARTS),
    )
    .expect("open");
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(list_entries.len(), 5);
    assert!(
        m.get_all_superfiles().is_empty(),
        "lazy mode leaves superfile_list.superfiles empty pending \
         the hierarchical query path; got {} superfiles",
        m.get_all_superfiles().len()
    );

    // Every part has an empty OnceCell.
    let n_loaded = list_entries
        .iter()
        .filter(|entry| m.get_cached_part_by_id(&entry.part_id).is_some())
        .count();
    assert_eq!(
        n_loaded, 0,
        "lazy mode must not have eager-fetched any parts; got {n_loaded} loaded"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_part_lazy_loads_on_first_access() {
    // Same setup as above (5 parts, forced-lazy). Calling
    // `ManifestSnapshot::part(id).await` on a specific part should
    // load exactly that one part. A second call on the
    // same part should be a OnceCell hit (no second
    // storage GET — verifiable by checking the OnceCell is
    // initialized AFTER the first call).
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let producer_opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_target_superfiles_per_part(TARGET_SUPERFILES_PER_PARTITION);
    let producer = Supertable::create(producer_opts).expect("create");
    for _i in 0..LAZY_MODE_PART_COUNT {
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
    }
    drop(producer);

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_BELOW_PARTS),
    )
    .expect("open");
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    let target_pid = list_entries[LAZY_LOAD_TARGET_PART_INDEX].part_id;

    // Pre-condition: target part's OnceCell empty.
    let part = m.get_cached_part_by_id(&target_pid);
    assert!(part.is_none(), "target part starts cold");
    drop(part);

    // First load: pulls bytes.
    let part = m.get_part_by_id(target_pid).await.expect("first load");
    assert_eq!(part.superfiles.len(), 1);

    // Cell is now populated.
    // Drop the DashMap `Ref` before any subsequent
    // `m.part(...).await` — that method takes a write lock
    // on the same shard via `entry()`, which would
    // deadlock against a still-held read `Ref`.
    {
        let _part = m
            .get_cached_part_by_id(&target_pid)
            .expect("still in cache");
    }

    // Other parts stay cold. Same shard-lock discipline:
    // each iteration's `Ref` drops at end of its closure
    // body.
    let other_loaded = list_entries
        .iter()
        .filter(|e| e.part_id != target_pid)
        .filter(|entry| m.get_cached_part_by_id(&entry.part_id).is_some())
        .count();
    assert_eq!(
        other_loaded, 0,
        "lazy-loading one part must not pull any others; got {other_loaded} other loaded"
    );

    // Second load on the same part: OnceCell hit.
    let part_again = m.get_part_by_id(target_pid).await.expect("second load");
    // Both references point at the same Arc — OnceCell
    // hands out an Arc::clone on each get_or_init call.
    assert!(
        Arc::ptr_eq(&part, &part_again),
        "second part().await must hit the OnceCell (same Arc)"
    );
}

#[test]
fn with_eager_load_threshold_zero_forces_lazy_on_tiny_manifest() {
    // Even a 1-part manifest goes lazy when threshold=0.
    // Useful for tests that want to exercise the lazy path
    // without producing many parts.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
    }

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY),
    )
    .expect("open");
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(list_entries.len(), 1);
    assert!(
        m.get_all_superfiles().is_empty(),
        "threshold=0 forces lazy even on 1-part manifest"
    );
    let part = m.get_cached_part_by_id(&list_entries[0].part_id);
    assert!(part.is_none(), "threshold=0 must not eager-fetch");
}
