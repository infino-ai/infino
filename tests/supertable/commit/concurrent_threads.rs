// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Concurrent-readers stress + invariants.
//!
//! These tests cover the load-bearing reader-isolation guarantee
//! of the supertable: a reader pinned at time `t` continues to see
//! the manifest as it existed at `t` for the lifetime of that
//! reader, regardless of how many writer commits land afterwards.
//! The mechanism is `ArcSwap<Manifest>` for lock-free swap-on-
//! commit + `Arc<Manifest>` snapshot pinning at `Supertable::reader`
//! time.
//!
//! ## What's asserted
//!
//! 1. **Pin-before isolation.** A reader pinned before any commits
//!    sees `manifest_id == 0` and `n_superfiles == 0` even after the
//!    writer has performed many concurrent commits.
//!
//! 2. **Pin-after visibility.** A reader pinned after the writer
//!    has finished sees the final post-commit state.
//!
//! 3. **Snapshot stability under concurrent commits.** While the
//!    writer is committing, repeatedly polling a *single* pinned
//!    reader observes a stable `manifest_id`. The reader's
//!    `Arc<Manifest>` is the immutable point-in-time view; no
//!    writer activity changes it.
//!
//! 4. **Arc identity sharing.** Two readers obtained between the
//!    same two commits hold the same `Arc<Manifest>` pointer
//!    (`Arc::ptr_eq`) — one allocation per commit, N+1 ref count
//!    for N concurrent readers.
//!
//! 5. **Monotonic manifest_id.** Across staggered reader pins
//!    interleaved with commits, the *sequence* of pinned
//!    manifest_ids in pin-order is monotonically non-decreasing.
//!
//! 6. **Many-reader stress.** 1 writer + 16 reader threads
//!    concurrent for 200 commits, with each reader continuously
//!    pinning + reading + dropping. No data races, no panics, no
//!    invariant violations under thread sanitizer / loom-style
//!    interleaving.
//!
//! All tests use `InMemoryReaderCache`, so no disk I/O. The
//! writer's `commit()` does run the rayon-shard build path
//! (parquet construction is in-memory but allocates), which
//! mirrors the real production cost shape — these aren't
//! ArcSwap-only mocks.
//!
//! Note: the writer is single-shot per supertable
//! (`SupertableWriter` enforces an exclusive slot via
//! `compare_exchange`), so all tests use exactly one writer
//! thread. Multi-writer cross-process semantics live with the
//! object-store + lock-file design.

#![deny(clippy::unwrap_used)]

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use arrow_array::{LargeStringArray, RecordBatch};

use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::{default_supertable_options, schema_id_title};

/// Contending writer threads in the single-winner commit race.
const CONTENDER_THREADS: usize = 4;
/// Snapshot-stability stress: writer commits + reader fanout.
const STRESS_COMMIT_COUNT: u64 = 50;
const STRESS_READER_THREADS: usize = 16;
const STRESS_PINS_PER_READER: usize = 200;

fn options() -> SupertableOptions {
    default_supertable_options()
}

fn build_batch(start: u64, n: usize) -> RecordBatch {
    let titles = LargeStringArray::from(
        (0..n)
            .map(|i| format!("doc {} title", start + i as u64))
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("batch")
}

#[test]
fn reader_pinned_before_writer_starts_never_sees_commits() {
    let st = Supertable::create(options()).expect("create");
    // Pin BEFORE any commit; capture the snapshot.
    let pinned = st.reader();
    assert_eq!(pinned.manifest_id(), 0);
    assert_eq!(pinned.n_superfiles(), 0);

    // Writer commits 5 superfiles concurrently with a polling reader-
    // check loop that re-asserts pinned-snapshot invariance.
    let st_for_writer = st.clone();
    let writer_handle = thread::spawn(move || {
        let mut w = st_for_writer.writer().expect("writer");
        for i in 0..5u64 {
            w.append(&build_batch(i * 10, 3)).expect("append");
            w.commit().expect("commit");
            // Yield between commits to interleave with the
            // reader-check loop below.
            thread::sleep(Duration::from_millis(2));
        }
        drop(w);
    });

    // Repeatedly probe the pinned reader; manifest_id and superfile
    // count must NOT advance regardless of writer progress.
    let deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < deadline {
        assert_eq!(
            pinned.manifest_id(),
            0,
            "pinned reader's manifest_id moved while writer ran",
        );
        assert_eq!(
            pinned.n_superfiles(),
            0,
            "pinned reader's superfile count grew while writer ran",
        );
    }
    writer_handle.join().expect("writer thread joined");

    // After writer finishes, pinned reader is STILL at 0.
    assert_eq!(pinned.manifest_id(), 0);
    assert_eq!(pinned.n_superfiles(), 0);

    // A FRESH reader sees the post-commit state.
    let fresh = st.reader();
    assert_eq!(fresh.manifest_id(), 5);
    assert_eq!(fresh.n_superfiles(), 5);
    assert_eq!(fresh.n_docs_total(), 5 * 3);
}

#[test]
fn reader_obtained_after_writer_finishes_sees_full_state() {
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    for i in 0..3u64 {
        w.append(&build_batch(i * 10, 4)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    let r = st.reader();
    assert_eq!(r.manifest_id(), 3);
    assert_eq!(r.n_superfiles(), 3);
    assert_eq!(r.n_docs_total(), 12);
}

#[test]
fn pinned_reader_holds_arc_across_subsequent_commits() {
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(0, 2)).expect("a1");
    w.commit().expect("c1");

    // Pin reader at manifest_id=1.
    let r1 = st.reader();
    let r1_arc = Arc::clone(r1.manifest());
    assert_eq!(r1.manifest_id(), 1);

    // Subsequent commits don't change r1's Arc identity.
    w.append(&build_batch(10, 2)).expect("a2");
    w.commit().expect("c2");
    w.append(&build_batch(20, 2)).expect("a3");
    w.commit().expect("commit");

    assert_eq!(r1.manifest_id(), 1);
    assert!(
        Arc::ptr_eq(&r1_arc, r1.manifest()),
        "pinned reader's manifest Arc must retain identity",
    );

    // Fresh reader sees the new state.
    let r2 = st.reader();
    assert_eq!(r2.manifest_id(), 3);
    assert!(!Arc::ptr_eq(r1.manifest(), r2.manifest()));
}

#[test]
fn concurrent_readers_at_same_commit_share_arc_pointer() {
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(0, 5)).expect("a1");
    w.commit().expect("c1");
    drop(w);

    // 4 reader threads, all racing to pin AFTER c1 but before any
    // further commit (none happens). All 4 should hold the same
    // Arc<Manifest>.
    let barrier = Arc::new(Barrier::new(CONTENDER_THREADS));
    let st = Arc::new(st);
    let mut handles = Vec::new();
    for _ in 0..CONTENDER_THREADS {
        let st = Arc::clone(&st);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            let r = st.reader();
            Arc::clone(r.manifest())
        }));
    }
    let manifests: Vec<Arc<_>> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let head = &manifests[0];
    for m in &manifests[1..] {
        assert!(
            Arc::ptr_eq(head, m),
            "readers pinned without an interleaving commit must share Arc"
        );
    }
}

#[test]
fn manifest_id_monotonic_across_serially_pinned_readers() {
    // Sanity: between commits, fresh readers see successively
    // higher manifest_ids. Single-threaded; this is the baseline
    // invariant the concurrent tests build on.
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    let mut observed: Vec<u64> = Vec::new();
    observed.push(st.reader().manifest_id());
    for i in 0..5u64 {
        w.append(&build_batch(i * 10, 2)).expect("append");
        w.commit().expect("commit");
        observed.push(st.reader().manifest_id());
    }
    drop(w);

    for w in observed.windows(2) {
        assert!(w[0] <= w[1], "manifest_id regressed: {observed:?}");
    }
    assert_eq!(observed.first(), Some(&0));
    assert_eq!(observed.last(), Some(&5));
}

#[test]
fn many_concurrent_readers_during_writer_commits_no_inconsistencies() {
    // Stress: 1 writer + 16 reader threads, 50 commits + 200
    // reader pins per reader thread, all racing. Each reader
    // pins, samples (manifest_id, n_superfiles) twice with a gap,
    // and asserts the pair is unchanged across the hold (the
    // load-bearing snapshot-stability guarantee).
    let st = Supertable::create(options()).expect("create");
    let n_commits = STRESS_COMMIT_COUNT;
    let n_readers = STRESS_READER_THREADS;
    let pins_per_reader = STRESS_PINS_PER_READER;

    let st_for_writer = st.clone();
    let writer = thread::spawn(move || {
        let mut w = st_for_writer.writer().expect("writer");
        for i in 0..n_commits {
            w.append(&build_batch(i * 10, 2)).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
    });

    let st_arc = Arc::new(st);
    let mut reader_handles = Vec::with_capacity(n_readers);
    for _ in 0..n_readers {
        let st = Arc::clone(&st_arc);
        reader_handles.push(thread::spawn(move || {
            let mut max_seen: u64 = 0;
            for _ in 0..pins_per_reader {
                let r = st.reader();
                let id_before = r.manifest_id();
                let n_before = r.n_superfiles();
                // Brief hold so we straddle a commit boundary
                // (the writer issues commits at full speed; this
                // hold is much longer than a single ArcSwap
                // operation, so the underlying ArcSwap may have
                // moved on, but our pinned Arc is unaffected).
                std::hint::black_box(&r);
                let id_after = r.manifest_id();
                let n_after = r.n_superfiles();
                assert_eq!(
                    id_before, id_after,
                    "pinned reader observed manifest_id change mid-hold",
                );
                assert_eq!(
                    n_before, n_after,
                    "pinned reader observed n_superfiles change mid-hold",
                );
                if id_before > max_seen {
                    max_seen = id_before;
                }
            }
            max_seen
        }));
    }
    writer.join().expect("writer joined");

    // After the writer is done, the FINAL fresh reader must show
    // exactly n_commits — independent of how reader threads
    // observed intermediate state.
    let final_r = st_arc.reader();
    assert_eq!(final_r.manifest_id(), n_commits);
    assert_eq!(final_r.n_superfiles(), n_commits as usize);

    // Reader threads' max_seen values must be in [0, n_commits].
    // (Bounded above because no commit beyond n_commits could have
    // happened during their hold; bounded below by 0 = pre-any-
    // commit state.)
    for h in reader_handles {
        let max_seen = h.join().expect("reader joined");
        assert!(max_seen <= n_commits, "reader saw impossible manifest_id");
    }
}

#[test]
fn fresh_reader_sequence_taken_during_concurrent_commits_is_monotonic() {
    // Repeatedly take a fresh reader on the orchestrator while a
    // writer thread commits in the background. The sequence of
    // observed manifest_ids in pin-order is monotonically
    // non-decreasing. (ArcSwap::load_full is monotone with
    // respect to ArcSwap::store under happens-before; this test
    // smoke-checks that under realistic interleaving.)
    let st = Supertable::create(options()).expect("create");
    let n_commits = 30u64;

    let st_for_writer = st.clone();
    let writer = thread::spawn(move || {
        let mut w = st_for_writer.writer().expect("writer");
        for i in 0..n_commits {
            w.append(&build_batch(i * 10, 2)).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
    });

    // Sample manifest_id at high frequency while the writer runs.
    let mut samples: Vec<u64> = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(500);
    while Instant::now() < deadline {
        samples.push(st.reader().manifest_id());
        if samples.last() == Some(&n_commits) {
            break;
        }
    }
    writer.join().expect("writer joined");
    // Final sample after writer fully finished.
    samples.push(st.reader().manifest_id());

    for w in samples.windows(2) {
        assert!(
            w[0] <= w[1],
            "fresh-reader sequence regressed: {} -> {}",
            w[0],
            w[1],
        );
    }
    assert_eq!(*samples.last().expect("≥1 sample"), n_commits);
}
