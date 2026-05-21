//! Cross-supertable `_id` uniqueness under concurrent
//! minting.
//!
//! The supertable injects a 128-bit Snowflake-shaped id on
//! every `append()` via
//! `utils::idgen::IdGenerator`. Each `Supertable::create()` /
//! `::open()` constructs a fresh generator with a 40-bit
//! random worker_id; no coordination across supertables. This
//! file validates that property under two stress shapes:
//!
//! 1. **In-process: 16 generators × 100K ids.** Birthday-
//!    collision probability for 16 random 40-bit worker_ids
//!    is ≈ 1.1×10⁻¹⁰, so 100 runs without collision is the
//!    expected outcome — the test exercises the parallel-
//!    mint path, not a worst-case collision scenario. Mints
//!    directly via `IdGenerator` (not through the writer's
//!    commit path) so the test runs in milliseconds, not
//!    minutes.
//!
//! 2. **Cross-handle: 4 supertable handles sharing storage.**
//!    Each handle constructs its own
//!    `Supertable` against a shared LocalFs backend,
//!    appends + commits a small batch, and a 5th handle
//!    opens against the same storage and runs `SELECT _id
//!    FROM supertable` to verify global uniqueness of the
//!    committed corpus. Tests the full path through the
//!    auto-injection in `append()` + the OCC retry on the
//!    manifest pointer.

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use arrow_array::{LargeStringArray, RecordBatch};

use infino::supertable::Supertable;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::utils::idgen::IdGenerator;
use infino::test_helpers::{default_supertable_options, schema_id_title};
use tempfile::TempDir;

const STRESS_N_WORKERS: usize = 16;
const STRESS_IDS_PER_WORKER: usize = 100_000;

#[test]
fn stress_16_generators_each_100k_ids_all_globally_unique() {
    // Spawn N threads, each owning its own IdGenerator with
    // a freshly-randomized worker_id. Each thread mints
    // STRESS_IDS_PER_WORKER ids and returns them to the
    // orchestrator. The union must be `n_workers ×
    // ids_per_worker` distinct values.
    let handles: Vec<thread::JoinHandle<Vec<i128>>> = (0..STRESS_N_WORKERS)
        .map(|_| {
            thread::spawn(|| {
                let g = IdGenerator::new();
                (0..STRESS_IDS_PER_WORKER).map(|_| g.next_id()).collect()
            })
        })
        .collect();

    let mut all: HashSet<i128> = HashSet::with_capacity(STRESS_N_WORKERS * STRESS_IDS_PER_WORKER);
    for h in handles {
        let ids = h.join().expect("worker thread panicked");
        assert_eq!(ids.len(), STRESS_IDS_PER_WORKER);
        for id in ids {
            assert!(
                all.insert(id),
                "duplicate id across workers: {id} — birthday collision \
                 on the 40-bit random worker_id would be the most likely \
                 cause, expected probability ~1.1e-10 at 16 workers"
            );
        }
    }
    assert_eq!(all.len(), STRESS_N_WORKERS * STRESS_IDS_PER_WORKER);
}

#[test]
fn stress_two_generators_with_explicit_same_worker_id_still_unique_within_one_run() {
    // Sanity probe: even if two generators happen to share
    // the same worker_id (the catastrophic scenario the
    // 40-bit space is designed to make extremely unlikely),
    // the per-generator timestamp + sequence counter keeps
    // *within-generator* ids strictly monotonic. The test
    // doesn't claim cross-generator uniqueness in this case
    // — that's the whole point of the random worker_id.
    let g1 = IdGenerator::with_worker_id(0xABCD);
    let g2 = IdGenerator::with_worker_id(0xABCD);
    let n = 10_000;
    let ids1: Vec<i128> = (0..n).map(|_| g1.next_id()).collect();
    let ids2: Vec<i128> = (0..n).map(|_| g2.next_id()).collect();

    // Each individual run is strictly monotonic.
    for w in ids1.windows(2) {
        assert!(w[0] < w[1]);
    }
    for w in ids2.windows(2) {
        assert!(w[0] < w[1]);
    }
}

// ---------------------------------------------------------
// Cross-handle (multi-supertable) test.
// ---------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn four_handles_to_shared_storage_produce_globally_unique_ids() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    const N_HANDLES: usize = 4;
    const ROWS_PER_HANDLE: u64 = 100;

    // Each handle appends a small batch and commits. The
    // commits race on the storage's manifest pointer; OCC
    // retry inside `persist_commit` serializes them in some
    // order. The auto-injected `_id` values are minted by
    // each handle's own IdGenerator with its own random
    // worker_id.
    let mut tasks = Vec::with_capacity(N_HANDLES);
    for handle_idx in 0..N_HANDLES {
        let storage = Arc::clone(&storage);
        tasks.push(tokio::task::spawn_blocking(move || {
            let st = Supertable::create(default_supertable_options().with_storage(storage));
            let mut w = st.writer().expect("writer");
            let titles: Vec<String> = (0..ROWS_PER_HANDLE)
                .map(|i| format!("h{handle_idx}_doc{i}"))
                .collect();
            let titles_refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            let batch = RecordBatch::try_new(
                schema_id_title(),
                vec![Arc::new(LargeStringArray::from(titles_refs))],
            )
            .expect("batch");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }));
    }
    for t in tasks {
        t.await.expect("task");
    }

    // Open a fresh handle against the same storage and
    // inspect the manifest's per-segment `(id_min, id_max)`
    // ranges. Each handle's single commit produces exactly
    // one segment under the default single-threaded writer
    // pool; ids within a segment form a contiguous
    // monotonic block, so cross-handle uniqueness reduces
    // to "no two superfiles' ranges overlap." This avoids
    // pulling segment bytes back from storage just to
    // verify ids — the manifest already carries everything
    // we need.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .await
            .expect("open");
    let reader = consumer.reader();
    let segs = &reader.manifest().superfile_list.superfiles;
    assert_eq!(
        segs.len(),
        N_HANDLES,
        "expected one segment per handle; got {}",
        segs.len()
    );

    let mut ranges: Vec<(i128, i128)> = segs.iter().map(|s| (s.id_min, s.id_max)).collect();
    ranges.sort_by_key(|(lo, _)| *lo);

    for window in ranges.windows(2) {
        let (lo_a, hi_a) = window[0];
        let (lo_b, _) = window[1];
        assert!(
            hi_a < lo_b,
            "segment id ranges overlap: ({lo_a}, {hi_a}) vs ({lo_b}, _)"
        );
    }
    // Total id count across all superfiles matches.
    let total_rows: u64 = segs.iter().map(|s| s.n_docs).sum();
    assert_eq!(total_rows, N_HANDLES as u64 * ROWS_PER_HANDLE);
}
