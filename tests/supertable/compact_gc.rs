// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Compact + GC integration test.
//!
//! Verifies the full lifecycle:
//! 1. Multiple commits produce multiple superfiles on disk.
//! 2. BM25 queries return expected hits.
//! 3. Compaction merges the superfiles into one; stale files remain
//!    on disk until GC runs.
//! 4. GC (safety_gap = 0) deletes stale objects; only live files remain.
//! 5. Data remains fully queryable after GC.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, time::Duration};

use infino::{
    CompactionSettings, OptimizeOptions,
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

const TOP_K: usize = 10;

fn small_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        ..CompactionSettings::default()
    })
}

fn count_dir(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .count()
}

fn commit_titles(st: &Supertable, titles: &[&str]) {
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

#[test]
fn compact_then_gc_removes_stale_files_and_preserves_queries() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    // Ten commits so combined live_bytes exceed the compaction floor (~10 KiB).
    // Each commit is a unique first-word marker for post-GC query verification.
    let markers = [
        "alphatoken",
        "betatoken",
        "gammatoken",
        "deltatoken",
        "epsilontoken",
        "zetatoken",
        "etatoken",
        "thetatoken",
        "iotatoken",
        "kappatoken",
    ];
    for m in &markers {
        // Two filler docs alongside the unique marker so superfiles are
        // large enough to reach the compaction floor (~10 KiB combined).
        commit_titles(
            &st,
            &[&format!("{m} marker"), "filler alpha", "filler bravo"],
        );
    }

    let n_commits = markers.len();
    let data_dir = dir.path().join("data");
    let lists_dir = dir.path().join("manifest-lists");

    assert_eq!(
        count_dir(&data_dir),
        n_commits,
        "one superfile per commit before compact"
    );
    assert_eq!(
        count_dir(&lists_dir),
        n_commits,
        "one manifest list per commit before compact"
    );

    let r = st.reader();
    assert_eq!(r.n_superfiles(), n_commits);
    assert_eq!(r.n_docs_total(), (n_commits * 3) as u64);

    // Spot-check three markers are queryable.
    assert_eq!(
        r.bm25_hits("title", "alphatoken", TOP_K, BoolMode::Or)
            .expect("query alpha")
            .len(),
        1
    );
    assert_eq!(
        r.bm25_hits("title", "kappatoken", TOP_K, BoolMode::Or)
            .expect("query kappa")
            .len(),
        1
    );

    // Compact: all 10 superfiles merge into one (or a small number).
    st.optimize(&small_optimize_opts()).expect("optimize");

    let r = st.reader();
    let n_after_compact = r.n_superfiles();
    assert!(
        n_after_compact < n_commits,
        "superfile count must decrease after compaction: got {n_after_compact}"
    );
    assert_eq!(
        r.n_docs_total(),
        (n_commits * 3) as u64,
        "doc count preserved after compact"
    );

    // Stale superfiles still on disk before GC (old + new compacted).
    assert!(
        count_dir(&data_dir) > n_after_compact,
        "stale superfiles must still be on disk before GC"
    );

    // GC with zero safety gap — every non-live file is eligible.
    let report = st.gc(Duration::ZERO).expect("gc");
    assert!(report.objects_deleted > 0, "GC must delete stale objects");
    assert_eq!(report.delete_errors, 0, "no delete errors");

    // Only the compacted superfile(s) survive in data/.
    assert_eq!(
        count_dir(&data_dir),
        n_after_compact,
        "only compacted superfiles remain after GC"
    );
    // Only the current manifest list survives.
    assert_eq!(
        count_dir(&lists_dir),
        1,
        "only current manifest list remains after GC"
    );

    // All markers still queryable after GC.
    let r = st.reader();
    for m in &markers {
        assert_eq!(
            r.bm25_hits("title", m, TOP_K, BoolMode::Or)
                .expect("query after gc")
                .len(),
            1,
            "marker {m} not found after GC"
        );
    }
}
