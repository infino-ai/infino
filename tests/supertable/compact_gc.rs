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

use chrono::{Duration as ChronoDuration, Utc};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    CompactionSettings, GcSettings, OptimizeOptions,
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
        wal::{
            persistence::WalStore,
            state_doc::{
                OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
                WalStateDoc,
            },
        },
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
    let manifest_dir = dir.path().join("manifest");

    assert_eq!(
        count_dir(&data_dir),
        n_commits,
        "one superfile per commit before compact"
    );
    // One manifest per commit, plus the empty manifest `create` published
    // (manifest_id 0) before the first append.
    assert_eq!(
        count_dir(&manifest_dir),
        n_commits + 1,
        "one manifest per commit, plus create's empty manifest, before compact"
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
    // Only the current manifest survives.
    assert_eq!(
        count_dir(&manifest_dir),
        1,
        "only current manifest remains after GC"
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

#[test]
fn gc_reaps_tombstone_sidecar_for_merged_away_superfile() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

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
        commit_titles(
            &st,
            &[&format!("{m} marker"), "filler alpha", "filler bravo"],
        );
    }

    let mut w = st.writer().expect("writer");
    let predicate: Expr = col("title").eq(lit("alphatoken marker"));
    let pending = w.delete(predicate).expect("delete");
    assert_eq!(pending.matched, 1);
    w.commit().expect("commit delete");
    drop(w);

    let superfiles_dir = dir.path().join("superfiles");
    assert_eq!(
        count_dir(&superfiles_dir),
        1,
        "delete writes exactly one tombstone sidecar"
    );

    st.optimize(&small_optimize_opts()).expect("optimize");
    // Compaction seals every input's sidecar (even untouched ones), so all
    // 10 input superfiles now have a `.tombstones` file, all orphaned since
    // none of those superfiles are in the manifest anymore.
    assert_eq!(
        count_dir(&superfiles_dir),
        markers.len(),
        "sidecars for merged-away superfiles aren't reaped yet: \
         optimize's default gc safety gap is 24h"
    );

    let report = st.gc(Duration::ZERO).expect("gc");
    assert!(
        report.objects_deleted > 0,
        "gc must delete the orphaned sidecars"
    );
    assert_eq!(
        count_dir(&superfiles_dir),
        0,
        "orphaned tombstone sidecars reaped once their superfiles are gone from the manifest"
    );

    let r = st.reader();
    assert_eq!(
        r.bm25_hits("title", "alphatoken", TOP_K, BoolMode::Or)
            .expect("query alpha after gc")
            .len(),
        0,
        "deleted row stays gone"
    );
    for m in &markers[1..] {
        assert_eq!(
            r.bm25_hits("title", m, TOP_K, BoolMode::Or)
                .expect("query after gc")
                .len(),
            1,
            "marker {m} not found after GC"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn optimize_reaps_completed_wal_past_grace() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    commit_titles(&st, &["alpha", "bravo"]);

    let ws = WalStore::new(Arc::clone(&storage));
    let leftover = WalStateDoc {
        wal_id: WalId(1),
        schema_version: SCHEMA_VERSION,
        op_kind: OpKind::Delete,
        state: WalState::Complete,
        created_at: Utc::now() - ChronoDuration::minutes(10),
        lease: None,
        predicate_repr: "leftover from a crashed inline cleanup".into(),
        target_ids: vec![RowId(1)],
        new_row_count: None,
        new_row_content_hash: None,
        preallocated_superfile_id: None,
        minted_id_spans: Vec::new(),
        tombstone_progress: vec![TombstoneEntry {
            target_id: RowId(1),
            outcome: TombstoneOutcome::NotFound,
            tombstoned_in_superfile: None,
        }],
    };
    ws.create(&leftover).await.expect("seed leftover wal");

    let wal_dir = dir.path().join("wal").join("mutations");
    assert_eq!(count_dir(&wal_dir), 1, "leftover wal state doc seeded");

    st.optimize(&small_optimize_opts()).expect("optimize");

    assert_eq!(
        count_dir(&wal_dir),
        0,
        "optimize must reap a completed wal past its grace window"
    );
}

#[test]
fn optimize_honors_overridden_gc_safety_gap() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    // Ten commits so combined live_bytes clear the compaction floor, same
    // as `compact_then_gc_removes_stale_files_and_preserves_queries`.
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
        commit_titles(
            &st,
            &[&format!("{m} marker"), "filler alpha", "filler bravo"],
        );
    }

    let data_dir = dir.path().join("data");
    let n_commits = count_dir(&data_dir);

    // Default gc_safety_gap (1 day): the stale pre-compaction superfiles
    // were just written, so optimize()'s bundled gc sweep must not touch
    // them, even though compaction ran and left them orphaned.
    st.optimize(&small_optimize_opts())
        .expect("optimize with default gc_safety_gap");
    let n_after_compact = st.reader().n_superfiles();
    assert!(
        n_after_compact < n_commits,
        "compaction must have merged the ten commits into fewer superfiles"
    );
    assert_eq!(
        count_dir(&data_dir),
        n_commits + n_after_compact,
        "default gc_safety_gap must keep freshly orphaned superfiles on disk \
         alongside the newly compacted ones"
    );

    // Zeroing gc_safety_gap on the same table now reclaims them in the
    // same optimize() call, with no separate st.gc() needed.
    let opts = OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        ..CompactionSettings::default()
    })
    .with_gc(GcSettings::default().with_safety_gap(Duration::ZERO));
    st.optimize(&opts).expect("optimize with gc safety_gap=0");

    let r = st.reader();
    assert_eq!(
        count_dir(&data_dir),
        r.n_superfiles(),
        "gc_safety_gap=0 must reclaim every orphaned superfile down to the live set"
    );
}
