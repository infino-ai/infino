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
//! 6. GC drops the disk-cache copy of every superfile it deletes.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, path::Path, sync::Arc, time::Duration};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use chrono::{Duration as ChronoDuration, Utc};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    CompactionSettings, GcSettings, OptimizeOptions,
    superfile::{builder::FtsConfig, fts::reader::BoolMode},
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{LocalFsStorageProvider, StorageProvider},
        wal::{
            persistence::WalStore,
            state_doc::{
                OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
                WalStateDoc,
            },
        },
    },
    test_helpers::{
        build_title_batch, default_supertable_options, default_tokenizer, default_vector_config,
    },
};
use tempfile::TempDir;

const TOP_K: usize = 10;
/// Disk-cache budget for the drop-through tests: large enough that eviction
/// never fires, so anything missing from the cache was dropped by gc.
const CACHE_BUDGET_BYTES: u64 = 1 << 30;

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

/// Count tombstone sidecar objects only. LocalFS `put_if_match` drops an
/// advisory `superfiles/.lock` next to them; a raw directory count would
/// treat that lock as a sidecar and flake the post-optimize assertion.
fn count_tombstone_sidecars(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "tombstones"))
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

    let r = st.reader().expect("reader");
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

    let r = st.reader().expect("reader");
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
    let r = st.reader().expect("reader");
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
        count_tombstone_sidecars(&superfiles_dir),
        1,
        "delete writes exactly one tombstone sidecar"
    );

    st.optimize(&small_optimize_opts()).expect("optimize");
    // Compaction seals every input's sidecar (even untouched ones), so all
    // 10 input superfiles now have a `.tombstones` file, all orphaned since
    // none of those superfiles are in the manifest anymore.
    assert_eq!(
        count_tombstone_sidecars(&superfiles_dir),
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
        count_tombstone_sidecars(&superfiles_dir),
        0,
        "orphaned tombstone sidecars reaped once their superfiles are gone from the manifest"
    );

    let r = st.reader().expect("reader");
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
    let n_after_compact = st.reader().expect("reader").n_superfiles();
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

    let r = st.reader().expect("reader");
    assert_eq!(
        count_dir(&data_dir),
        r.n_superfiles(),
        "gc_safety_gap=0 must reclaim every orphaned superfile down to the live set"
    );
}

/// A table wired to its own disk cache, the shape both drop-through tests need:
/// commits warm-fill this cache, so gc has copies to drop.
fn table_with_disk_cache(
    options: SupertableOptions,
    storage: Arc<dyn StorageProvider>,
    cache_root: &Path,
) -> (Supertable, Arc<DiskCacheStore>) {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    let cache = DiskCacheStore::new(Arc::clone(&storage), cfg, pinned).expect("cache");
    let table = Supertable::create(
        options
            .with_storage(storage)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create");
    (table, cache)
}

/// Count promoted superfile copies in a disk-cache root (ignores `.tmp`
/// in-flight files and block sidecars).
fn count_cached_superfiles(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .expect("readdir cache")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".sf.parquet"))
        .count()
}

#[test]
fn gc_drops_disk_cache_copies_of_deleted_superfiles() {
    // 1. Ten commits warm-fill the cache with the superfiles they publish.
    // 2. Compaction supersedes those superfiles; GC deletes them from storage.
    // 3. The cache must end up holding no more than the live set, not dead
    //    copies waiting on budget pressure.
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let (st, cache) =
        table_with_disk_cache(default_supertable_options(), storage, cache_dir.path());

    for i in 0..10 {
        let t1 = format!("cache doc alpha {i}");
        let t2 = format!("cache doc beta {i}");
        commit_titles(&st, &[t1.as_str(), t2.as_str()]);
    }
    let cached_before = count_cached_superfiles(cache_dir.path());
    assert!(
        cached_before >= 10,
        "writer warm-inserts each commit's superfiles: {cached_before}"
    );

    st.optimize(&small_optimize_opts()).expect("optimize");
    let report = st.gc(Duration::ZERO).expect("gc");
    assert!(report.objects_deleted > 0, "superseded fragments reclaimed");

    // Cache converged: dead copies dropped alongside their storage objects,
    // live superfiles (and only those) may remain cached.
    let data_dir = dir.path().join("data");
    let cached_after = count_cached_superfiles(cache_dir.path());
    assert!(
        cached_after <= count_dir(&data_dir),
        "cache holds no more superfiles than storage: {cached_after} vs {}",
        count_dir(&data_dir)
    );
    assert!(
        cache.stats().n_gc_drops > 0,
        "drops are visible in cache stats"
    );

    // Data still fully queryable through the converged cache.
    let hits = st
        .bm25_search(
            "title",
            "alpha",
            TOP_K,
            BoolMode::Or,
            Default::default(),
            None,
        )
        .expect("bm25 after gc");
    assert!(!hits.is_empty(), "queries survive the drop-through");
}

/// Dimension for the hidden-index fixture; matches `default_vector_config`.
const HIDDEN_DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const HIDDEN_ROT_SEED: u64 = 37;
/// Commit + drain rounds. The hidden profile merges a cell on any two shards,
/// so two rounds already leave superseded inputs for gc to reclaim.
const HIDDEN_ROUNDS: usize = 3;

fn hidden_vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                HIDDEN_DIM as i32,
            ),
            false,
        ),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", HIDDEN_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

/// One-hot rows: doc `i` points at dim `i`, so every round lands in the same
/// cells and the hidden index accumulates shards there.
fn hidden_batch(schema: Arc<Schema>, round: usize) -> RecordBatch {
    let titles: Vec<String> = (0..HIDDEN_DIM)
        .map(|i| format!("hidden doc {round} {i}"))
        .collect();
    let mut flat = Vec::<f32>::with_capacity(HIDDEN_DIM * HIDDEN_DIM);
    for i in 0..HIDDEN_DIM {
        for d in 0..HIDDEN_DIM {
            flat.push(if d == i { 1.0 } else { 0.0 });
        }
    }
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let embeddings = FixedSizeListArray::try_new(
        item,
        HIDDEN_DIM as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .expect("fixed list");
    let cols: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(
            titles.iter().map(String::as_str).collect::<Vec<_>>(),
        )),
        Arc::new(embeddings),
    ];
    RecordBatch::try_new(schema, cols).expect("batch")
}

#[test]
fn gc_on_the_hidden_vector_index_drops_its_cache_copies() {
    // The hidden vector index shares the user table's cache but sweeps through
    // its own prefixed storage provider, so it needs its own proof:
    // 1. Each round commits rows and drains them into hidden cells, so a cell
    //    accumulates shards across rounds. The explicit drain matters: without
    //    it the whole corpus drains in one pass and nothing is superseded.
    // 2. Compaction merges each cell (the hidden profile merges on two shards).
    // 3. The hidden GC deletes the superseded shards, and must drop their cache
    //    copies. Counted after the user sweep, so the drops are its own.
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let (st, cache) = table_with_disk_cache(hidden_vector_options(), storage, cache_dir.path());

    // Drain per round so each round adds its own shards to the same cells;
    // without the explicit drain the whole corpus lands in one pass and
    // nothing is ever superseded.
    let schema = st.options().schema.clone();
    for round in 0..HIDDEN_ROUNDS {
        let mut w = st.writer().expect("writer");
        w.append(&hidden_batch(Arc::clone(&schema), round))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");
    }

    let prefix = st
        .vector_index_storage_prefix()
        .expect("hidden prefix")
        .to_owned();
    let hidden_data = dir.path().join(&prefix).join("data");
    let hidden_before = count_dir(&hidden_data);

    // Compact both tables, then sweep the user table first so anything the
    // hidden sweep drops is attributable to it alone.
    st.optimize(&small_optimize_opts()).expect("optimize");
    st.gc(Duration::ZERO).expect("user gc");
    let drops_before_hidden = cache.stats().n_gc_drops;

    let hidden = st.vector_index_table().expect("hidden index table");
    let hidden_report = hidden.gc(Duration::ZERO).expect("hidden gc");
    assert!(
        count_dir(&hidden_data) < hidden_before,
        "hidden compaction + gc must reclaim superseded shards ({} -> {})",
        hidden_before,
        count_dir(&hidden_data)
    );
    assert!(
        hidden_report.objects_deleted > 0,
        "hidden objects reclaimed"
    );
    assert!(
        cache.stats().n_gc_drops > drops_before_hidden,
        "the hidden sweep must drop its own cache copies (before={drops_before_hidden}, after={})",
        cache.stats().n_gc_drops
    );
}
