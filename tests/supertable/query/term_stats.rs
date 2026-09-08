// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Global term-stats sidecar behavior over the public surface: optimize
//! publishes it, global-stats queries read df from it instead of fanning
//! the query-time gather, appends compose (sidecar + uncovered tail),
//! and a maintenance republish reflects membership changes — with
//! ranking always equal to a never-optimized control table holding the
//! same content.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, time::Duration};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    CompactionSettings, OptimizeOptions,
    runtime_metrics::op_stats::with_op_stats,
    storage::StorageProvider,
    superfile::{
        builder::FtsConfig,
        fts::reader::{Bm25Stats, BoolMode},
    },
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore},
        storage::LocalFsStorageProvider,
    },
};
use tempfile::TempDir;

/// Docs per committed segment; every scored term keeps df ≥ 2 per
/// segment so nothing rides the inline (df=1) dictionary slot.
const DOCS_PER_SEGMENT: usize = 40;
/// Deterministic 2-shard builds.
const RAYON_POOL_THREADS: usize = 2;
/// k large enough that no assertion depends on ranking cutoffs.
const TOP_K: usize = 128;

/// Optimize settings whose compaction is a NO-OP (fill floor at 100%
/// keeps every superfile), so `optimize` degenerates to the maintenance
/// passes — in particular the term-stats refresh — without changing the
/// superfile layout. That keeps the table genuinely fragmented, which is
/// the shape where the sidecar earns its keep.
fn stats_only_optimize() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        min_fill_percent: 100,
        min_superfiles_for_merge: u64::MAX,
        ..CompactionSettings::default()
    })
}

fn schema_title() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

fn options_with_storage(dir: &TempDir) -> SupertableOptions {
    let storage = Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
    options_with_storage_provider(storage)
}

fn options_with_storage_provider(storage: Arc<dyn StorageProvider>) -> SupertableOptions {
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(schema_title(), vec![FtsConfig::new("title")], Vec::new())
        .expect("valid options")
        .with_writer_pool(writer_pool)
        .with_storage(storage)
}

/// One segment's titles: uniform 4-token docs so per-superfile `avgdl`
/// is layout-independent and global idf is the only ranking variable.
/// `alpha` df rises with `segment` so cross-segment df genuinely
/// matters to the sidecar sums.
fn segment_titles(segment: usize) -> Vec<String> {
    (0..DOCS_PER_SEGMENT)
        .map(|i| {
            let topic = if i % (segment + 2) == 0 {
                "alpha"
            } else {
                "beta"
            };
            let band = ["red", "green"][i % 2];
            format!("{topic} shared {band} s{segment}d{i:02}")
        })
        .collect()
}

fn commit_segment(st: &Supertable, segment: usize) {
    let titles = segment_titles(segment);
    let arr: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema_title(), vec![arr]).expect("batch");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
}

/// Ranked `(title, score)` rows for a global-stats BM25 query.
fn global_hits(st: &Supertable, query: &str, mode: BoolMode) -> Vec<(String, f32)> {
    let batches = st
        .reader()
        .expect("reader")
        .bm25_search(
            "title",
            query,
            TOP_K,
            mode,
            Bm25Stats::Global,
            Some(&["title", "score"]),
        )
        .expect("bm25_search");
    let mut out = Vec::new();
    for b in batches {
        let titles = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        let scores = b
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("score col");
        for i in 0..b.num_rows() {
            out.push((titles.value(i).to_string(), scores.value(i)));
        }
    }
    out
}

/// Planned read ranges for one scoped BM25 query.
fn planned_ranges(st: &Supertable, query: &str, mode: BoolMode, stats: Bm25Stats) -> u64 {
    let (hits, op) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits_stats("title", query, TOP_K, mode, stats)
            .expect("bm25")
    });
    assert!(!hits.is_empty(), "fixture query {query:?} must match");
    op.planned_read_ranges
}

/// The whole lifecycle in one narrative: publish, plan-parity, append
/// tail, republish — ranking always equal to the never-optimized
/// control.
#[test]
fn sidecar_covers_fragmented_table_and_composes_with_tail() {
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(options_with_storage(&dir)).expect("create");
    let ctrl_dir = TempDir::new().expect("tempdir");
    let control = Supertable::create(options_with_storage(&ctrl_dir)).expect("create control");
    for segment in 0..2 {
        commit_segment(&st, segment);
        commit_segment(&control, segment);
    }
    let n_before = st.reader().expect("reader").n_superfiles();
    assert!(
        n_before >= 2,
        "fixture must stay fragmented; got {n_before}"
    );

    // Stats-only optimize: same layout, sidecar published.
    st.optimize(&stats_only_optimize()).expect("optimize");
    assert_eq!(
        st.reader().expect("reader").n_superfiles(),
        n_before,
        "stats-only optimize must not change the layout"
    );

    // Ranking parity with the never-optimized control (which computes df
    // through the query-time gather): the sidecar sums must be exactly
    // the gather's sums.
    for (query, mode) in [
        ("alpha", BoolMode::Or),
        ("alpha shared", BoolMode::And),
        ("beta red", BoolMode::Or),
    ] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "sidecar-served ranking must equal the gather-served control for {query:?}"
        );
    }

    // Plan parity: with every superfile covered, a first global query —
    // AND shapes included, whose gather previously paid a dict-only
    // residual — plans EXACTLY the per-superfile work. (`red` appears
    // only in half the docs, so the AND prune and presence set genuinely
    // differ across shards.)
    let and_q = "alpha red";
    let per_superfile = planned_ranges(&st, and_q, BoolMode::And, Bm25Stats::PerSuperfile);
    assert_eq!(
        planned_ranges(&st, and_q, BoolMode::And, Bm25Stats::Global),
        per_superfile,
        "sidecar-covered global AND query plans the per-superfile work"
    );
    let or_q = "beta";
    let per_superfile_or = planned_ranges(&st, or_q, BoolMode::Or, Bm25Stats::PerSuperfile);
    assert_eq!(
        planned_ranges(&st, or_q, BoolMode::Or, Bm25Stats::Global),
        per_superfile_or,
        "sidecar-covered global OR query plans the per-superfile work"
    );

    // Appends carry the sidecar and compose with the uncovered tail:
    // segment 2 raises `alpha`'s corpus df, so global idf must reflect
    // sidecar + tail together — again equal to the control's gather.
    commit_segment(&st, 2);
    commit_segment(&control, 2);
    for (query, mode) in [("alpha", BoolMode::Or), ("alpha green", BoolMode::And)] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "sidecar + tail must equal the gather-served control for {query:?}"
        );
    }

    // A fresh maintenance pass re-covers the tail; plan parity returns
    // for a term the earlier queries never cached.
    st.optimize(&stats_only_optimize()).expect("re-optimize");
    let fresh_q = "green";
    let per_superfile_fresh = planned_ranges(&st, fresh_q, BoolMode::Or, Bm25Stats::PerSuperfile);
    assert_eq!(
        planned_ranges(&st, fresh_q, BoolMode::Or, Bm25Stats::Global),
        per_superfile_fresh,
        "re-covered table plans the per-superfile work on a fresh term"
    );
}

/// A merging optimize removes superfiles: the carry rule drops the old
/// reference inside the membership commit and the maintenance pass
/// republishes over the merged layout — ranking must stay equal to a
/// control that never optimized (same corpus, global stats are
/// layout-independent for uniform-length docs).
#[test]
fn merging_optimize_republishes_over_new_membership() {
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(options_with_storage(&dir)).expect("create");
    let ctrl_dir = TempDir::new().expect("tempdir");
    let control = Supertable::create(options_with_storage(&ctrl_dir)).expect("create control");
    for segment in 0..3 {
        commit_segment(&st, segment);
        commit_segment(&control, segment);
    }
    // Publish over the fragmented layout first, then merge for real.
    st.optimize(&stats_only_optimize()).expect("stats optimize");
    let merging = OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        // The fixture holds a handful of tiny superfiles; trip the
        // fragment-count trigger so the merge actually fires.
        min_superfiles_for_merge: 2,
        ..CompactionSettings::default()
    });
    st.optimize(&merging).expect("merging optimize");
    assert!(
        st.reader().expect("reader").n_superfiles()
            < control.reader().expect("reader").n_superfiles(),
        "merging optimize must have compacted (st {} vs control {})",
        st.reader().expect("reader").n_superfiles(),
        control.reader().expect("reader").n_superfiles()
    );
    for (query, mode) in [
        ("alpha", BoolMode::Or),
        ("alpha shared", BoolMode::And),
        ("beta green", BoolMode::Or),
    ] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "post-merge sidecar ranking must equal the control for {query:?}"
        );
    }
}

// ---- maintenance must not warm the disk cache -----------------------
//
// The stats pass reads dictionaries and df headers only. Opening the
// superfiles with background fills enabled would copy EVERY superfile
// into the attached disk cache — including compaction's fresh multi-GiB
// outputs — and on real object storage those fills outlive `optimize`
// and their reads bleed into whatever is measured next. The gate: after
// a stats-only optimize on a fill-capable cache, the cache holds only
// the chunks the dictionary walks actually touched, a small fraction of
// the table.

/// Bytes under a directory tree.
fn dir_bytes(root: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

/// Docs per segment in the cache-hygiene fixture — enough bytes per
/// superfile that a whole-file background fill is unmistakably larger
/// than the dictionary/header chunks the stats pass legitimately reads.
const CACHE_FIXTURE_DOCS_PER_SEGMENT: usize = 8_000;
/// Segments in the cache-hygiene fixture.
const CACHE_FIXTURE_SEGMENTS: usize = 3;
/// Settle bound for any background fills the maintenance pass spawned.
const FILL_SETTLE_TIMEOUT: Duration = Duration::from_secs(120);
/// The stats pass may cache the chunks it reads (dictionaries, df
/// headers); a fill copies whole superfiles. Half the stored bytes
/// separates the two regimes with wide margin on both sides.
const CACHE_BYTES_CEILING_FRACTION: f64 = 0.5;

#[test]
fn stats_only_optimize_does_not_fill_the_disk_cache() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));

    // Produce a fragmented table without any cache attached.
    let producer =
        Supertable::create(options_with_storage_provider(Arc::clone(&storage))).expect("create");
    for segment in 0..CACHE_FIXTURE_SEGMENTS {
        let titles: Vec<String> = (0..CACHE_FIXTURE_DOCS_PER_SEGMENT)
            .map(|i| {
                format!(
                    "alpha shared s{segment}d{i:05} \
                     filler{:04} filler{:04} filler{:04} filler{:04}",
                    i % 977,
                    (i * 7) % 977,
                    (i * 31) % 977,
                    (i * 131) % 977,
                )
            })
            .collect();
        let arr: ArrayRef = Arc::new(LargeStringArray::from(
            titles.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema_title(), vec![arr]).expect("batch");
        let mut w = producer.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(producer);
    let stored_bytes = dir_bytes(&dir.path().join("data"));
    assert!(
        stored_bytes > 0,
        "fixture must have persisted superfile bytes"
    );

    // Consumer with a fill-capable disk cache runs the maintenance.
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        disk_budget_bytes: 4 * stored_bytes,
        cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
        verify_crc_on_open: false,
        ..Default::default()
    };
    let cache = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg).expect("disk cache");
    let consumer = Supertable::open(options_with_storage_provider(storage).with_disk_cache(cache))
        .expect("open consumer");
    consumer
        .optimize(&stats_only_optimize())
        .expect("stats-only optimize");
    // Settle anything the pass spawned, so a regression cannot hide
    // behind a fill that is merely still running when we measure.
    consumer
        .wait_until_warm(FILL_SETTLE_TIMEOUT)
        .expect("fills settle");

    let cache_bytes = dir_bytes(cache_dir.path());
    let ceiling = (stored_bytes as f64 * CACHE_BYTES_CEILING_FRACTION) as u64;
    assert!(
        cache_bytes < ceiling,
        "stats-only optimize warmed the disk cache like a query would: \
         {cache_bytes} cached bytes vs {stored_bytes} stored (ceiling {ceiling}) — \
         the maintenance open must not spawn background fills"
    );
}

// ---- mutations on a covered table ----------------------------------
//
// Deletes and updates never rewrite a superfile's dictionary, so the df
// a covered superfile contributed to the sidecar stays exactly what it
// was — "gross" df, unchanged until a compaction rebuilds that
// dictionary. Both tests below assert the same invariant the append
// tests do (sidecar-served ranking equals a never-optimized control,
// which derives df from the query-time gather), because that is what
// makes the sidecar faithful rather than merely fast. Each then pins
// the gross semantics themselves, so a future change that quietly made
// mutations net-of-tombstones would fail here rather than surface as a
// scoring drift.

/// Titles in `segment` whose topic token is `alpha` — the rows the
/// mutation tests target, chosen because `alpha` df varies per segment
/// (see [`segment_titles`]) so its idf genuinely depends on the sums.
fn alpha_titles(segment: usize) -> Vec<String> {
    segment_titles(segment)
        .into_iter()
        .filter(|t| t.starts_with("alpha "))
        .collect()
}

/// The same titles with the topic token rewritten to `gamma`, keeping
/// the 4-token shape so per-superfile `avgdl` is unchanged and idf stays
/// the only ranking variable.
fn gamma_rewrites(titles: &[String]) -> Vec<String> {
    titles
        .iter()
        .map(|t| t.replacen("alpha ", "gamma ", 1))
        .collect()
}

fn title_batch(titles: &[String]) -> RecordBatch {
    let arr: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema_title(), vec![arr]).expect("batch")
}

fn in_list_predicate(titles: &[String]) -> Expr {
    col("title").in_list(titles.iter().map(|t| lit(t.as_str())).collect(), false)
}

/// Segments in the mutation fixtures: enough that the mutated segment is
/// a minority of the corpus, so a df that wrongly went net would move
/// scores measurably instead of marginally.
const MUTATION_SEGMENTS: usize = 3;

/// A delete on a covered table hides its rows but leaves every surviving
/// document's score untouched: the tombstone removes the row from result
/// sets while its contribution stays in both the sidecar's df sums and
/// the table's document count.
#[test]
fn delete_on_a_covered_table_leaves_surviving_scores_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(options_with_storage(&dir)).expect("create");
    let ctrl_dir = TempDir::new().expect("tempdir");
    let control = Supertable::create(options_with_storage(&ctrl_dir)).expect("create control");
    for segment in 0..MUTATION_SEGMENTS {
        commit_segment(&st, segment);
        commit_segment(&control, segment);
    }
    st.optimize(&stats_only_optimize()).expect("optimize");

    // Ranking before the delete, over a shape whose idf depends on
    // `alpha`'s corpus-wide df.
    let before = global_hits(&st, "alpha shared", BoolMode::Or);
    assert!(!before.is_empty(), "fixture query must match");

    // Delete the same rows from both tables.
    let targets = alpha_titles(0);
    assert!(
        !targets.is_empty(),
        "fixture must have alpha rows to delete"
    );
    let deleted = st
        .delete(in_list_predicate(&targets))
        .expect("delete on covered table");
    assert_eq!(deleted.matched(), targets.len());
    control
        .delete(in_list_predicate(&targets))
        .expect("delete on control");

    let after = global_hits(&st, "alpha shared", BoolMode::Or);
    assert_eq!(
        after,
        global_hits(&control, "alpha shared", BoolMode::Or),
        "sidecar-served ranking must equal the gather-served control after a delete"
    );

    // The deleted rows are gone from the result set...
    let target_set: std::collections::HashSet<&String> = targets.iter().collect();
    for (title, _) in &after {
        assert!(
            !target_set.contains(title),
            "deleted row {title:?} resurfaced"
        );
    }
    // ...and every survivor keeps the exact score it had before, which is
    // only true while df and the document count both stay gross. A df
    // that dropped with the tombstones would raise idf and move all of
    // these.
    assert!(
        !after.is_empty() && after.len() < before.len(),
        "delete must have removed some rows and left others to compare ({} then {})",
        before.len(),
        after.len()
    );
    for row in &after {
        assert!(
            before.contains(row),
            "surviving row {:?} changed score after the delete ({:?} not in the pre-delete \
             ranking) — df or the document count went net-of-tombstones",
            row.0,
            row
        );
    }
}

/// An update is a delete plus an insert, so on a covered table the
/// superseded row's df stays in the sidecar's sums (its dictionary is
/// untouched) while the replacement row's df arrives through the
/// uncovered tail — the term is counted in both, and the gather-served
/// control agrees exactly. A merging optimize is what finally rewrites
/// those dictionaries and converges the table onto net statistics.
#[test]
fn update_on_a_covered_table_counts_superseded_and_replacement_rows() {
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(options_with_storage(&dir)).expect("create");
    let ctrl_dir = TempDir::new().expect("tempdir");
    let control = Supertable::create(options_with_storage(&ctrl_dir)).expect("create control");
    for segment in 0..MUTATION_SEGMENTS {
        commit_segment(&st, segment);
        commit_segment(&control, segment);
    }
    st.optimize(&stats_only_optimize()).expect("optimize");

    // Rewrite segment 0's `alpha` rows to `gamma`: afterwards `alpha`
    // survives only in the later segments and `gamma` exists only in the
    // tail, so both directions of the sum are exercised.
    let targets = alpha_titles(0);
    let replacements = gamma_rewrites(&targets);
    let new_rows = title_batch(&replacements);
    let updated = st
        .update(in_list_predicate(&targets), &new_rows)
        .expect("update on covered table");
    assert_eq!(updated.matched(), targets.len());
    control
        .update(in_list_predicate(&targets), &new_rows)
        .expect("update on control");

    // Sidecar (covered segments) + tail (the replacement rows) must sum
    // to exactly what the control's gather computes, for a term left
    // only in the covered part, a term living only in the tail, and a
    // term spanning both.
    for (query, mode) in [
        ("alpha", BoolMode::Or),
        ("gamma", BoolMode::Or),
        ("shared", BoolMode::Or),
        ("gamma shared", BoolMode::And),
    ] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "sidecar + tail must equal the gather-served control for {query:?} after an update"
        );
    }

    // The superseded rows are invisible to results even though their
    // statistics remain.
    let target_set: std::collections::HashSet<&String> = targets.iter().collect();
    for (title, _) in global_hits(&st, "alpha shared", BoolMode::Or) {
        assert!(
            !target_set.contains(&title),
            "superseded row {title:?} resurfaced after the update"
        );
    }

    // Gross, not net: a table built from scratch with the post-update
    // content holds neither the superseded rows' df nor their document
    // count, so its idf — and therefore its scores — differ. This is the
    // intended semantics (matching the query-time gather, and matching
    // how deletes behave until compaction), so it is pinned rather than
    // left to drift.
    let fresh_dir = TempDir::new().expect("tempdir");
    let fresh = Supertable::create(options_with_storage(&fresh_dir)).expect("create fresh");
    for segment in 0..MUTATION_SEGMENTS {
        let titles: Vec<String> = match segment {
            0 => segment_titles(0)
                .into_iter()
                .map(|t| t.replacen("alpha ", "gamma ", 1))
                .collect(),
            other => segment_titles(other),
        };
        let mut w = fresh.writer().expect("writer");
        w.append(&title_batch(&titles)).expect("append");
        w.commit().expect("commit");
    }
    let mutated_scores: Vec<f32> = global_hits(&st, "gamma", BoolMode::Or)
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    let fresh_scores: Vec<f32> = global_hits(&fresh, "gamma", BoolMode::Or)
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    assert_eq!(
        mutated_scores.len(),
        fresh_scores.len(),
        "both tables must return the same `gamma` rows, so the comparison below is about \
         scores rather than result-set size"
    );
    assert!(
        !mutated_scores.is_empty(),
        "fixture must return replacement rows to compare"
    );
    assert_ne!(
        mutated_scores, fresh_scores,
        "an updated table scored identically to a from-scratch rebuild — the superseded \
         rows' statistics were dropped before compaction rewrote their dictionaries"
    );
}
