// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 ranking stays put across deletes, updates and a merging optimize:
//! a table that ran maintenance ranks exactly like a never-optimized control
//! holding the same content, and document frequency stays gross (tombstoned
//! rows still count) until compaction rewrites the dictionaries.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{ArrayRef, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    Bm25SearchOptions, CompactionSettings, OptimizeOptions,
    superfile::{builder::FtsConfig, fts::reader::BoolMode},
    supertable::{Supertable, SupertableOptions, storage::LocalFsStorageProvider},
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Docs per committed segment; every scored term keeps df ≥ 2 per
/// segment so nothing rides the inline (df=1) dictionary slot.
const DOCS_PER_SEGMENT: usize = 40;
/// Deterministic 2-shard builds.
const RAYON_POOL_THREADS: usize = 2;
/// k large enough that no assertion depends on ranking cutoffs.
const TOP_K: usize = 128;
/// Segments per fixture: enough that the mutated segment is a minority of
/// the corpus, so a df that wrongly went net would move scores measurably.
const SEGMENTS: usize = 3;

/// Optimize that merges nothing, so only the maintenance passes run.
fn maintenance_only_optimize() -> OptimizeOptions {
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
    let writer_pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(schema_title(), vec![FtsConfig::new("title")], Vec::new())
        .expect("valid options")
        .with_writer_pool(writer_pool)
        .with_storage(storage)
}

/// One segment's titles: uniform 4-token docs so per-superfile `avgdl` is
/// layout-independent and global idf is the only ranking variable. `alpha`
/// df rises with `segment`, so cross-segment df genuinely matters.
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

fn title_batch(titles: &[String]) -> RecordBatch {
    let arr: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema_title(), vec![arr]).expect("batch")
}

fn commit_titles(st: &Supertable, titles: &[String]) {
    let mut w = st.writer().expect("writer");
    w.append(&title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

/// A maintained table and a never-optimized control, both holding
/// `SEGMENTS` segments.
fn maintained_and_control() -> (TempDir, Supertable, TempDir, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(options_with_storage(&dir)).expect("create");
    let ctrl_dir = TempDir::new().expect("tempdir");
    let control = Supertable::create(options_with_storage(&ctrl_dir)).expect("create control");
    for segment in 0..SEGMENTS {
        commit_titles(&st, &segment_titles(segment));
        commit_titles(&control, &segment_titles(segment));
    }
    st.optimize(&maintenance_only_optimize())
        .expect("maintenance optimize");
    (dir, st, ctrl_dir, control)
}

/// Ranked `(title, score)` rows for a BM25 query.
fn global_hits(st: &Supertable, query: &str, mode: BoolMode) -> Vec<(String, f32)> {
    let batches = st
        .reader()
        .expect("reader")
        .bm25_search(
            "title",
            query,
            TOP_K,
            Bm25SearchOptions::new().with_mode(mode),
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
            .downcast_ref::<Float32Array>()
            .expect("score col");
        for i in 0..b.num_rows() {
            out.push((titles.value(i).to_string(), scores.value(i)));
        }
    }
    out
}

/// Titles in `segment` whose topic token is `alpha`: the rows the mutation
/// tests target, since `alpha`'s idf depends on every segment's df.
fn alpha_titles(segment: usize) -> Vec<String> {
    segment_titles(segment)
        .into_iter()
        .filter(|t| t.starts_with("alpha "))
        .collect()
}

/// The same titles with `alpha` rewritten to `gamma`, keeping the 4-token
/// shape so `avgdl` is unchanged and idf stays the only ranking variable.
fn gamma_rewrites(titles: &[String]) -> Vec<String> {
    titles
        .iter()
        .map(|t| t.replacen("alpha ", "gamma ", 1))
        .collect()
}

fn in_list_predicate(titles: &[String]) -> Expr {
    col("title").in_list(titles.iter().map(|t| lit(t.as_str())).collect(), false)
}

/// A merging optimize changes the superfiles but not the ranking: it must
/// equal a control that never optimized.
#[test]
fn merging_optimize_keeps_ranking() {
    let (_dir, st, _ctrl_dir, control) = maintained_and_control();
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
        "merging optimize must have compacted"
    );
    for (query, mode) in [
        ("alpha", BoolMode::Or),
        ("alpha shared", BoolMode::And),
        ("beta green", BoolMode::Or),
    ] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "post-merge ranking must equal the control for {query:?}"
        );
    }
}

/// A delete hides its rows but leaves every survivor's score untouched: the
/// deleted rows still count in df and in the document count until
/// compaction.
#[test]
fn delete_leaves_surviving_scores_unchanged() {
    let (_dir, st, _ctrl_dir, control) = maintained_and_control();
    let before = global_hits(&st, "alpha shared", BoolMode::Or);
    assert!(!before.is_empty(), "fixture query must match");

    let targets = alpha_titles(0);
    assert!(
        !targets.is_empty(),
        "fixture must have alpha rows to delete"
    );
    let deleted = st.delete(in_list_predicate(&targets)).expect("delete");
    assert_eq!(deleted.matched(), targets.len());
    control
        .delete(in_list_predicate(&targets))
        .expect("delete on control");

    let after = global_hits(&st, "alpha shared", BoolMode::Or);
    assert_eq!(
        after,
        global_hits(&control, "alpha shared", BoolMode::Or),
        "ranking must equal the control after a delete"
    );
    let target_set: HashSet<&String> = targets.iter().collect();
    for (title, _) in &after {
        assert!(
            !target_set.contains(title),
            "deleted row {title:?} resurfaced"
        );
    }
    assert!(
        !after.is_empty() && after.len() < before.len(),
        "delete must have removed some rows and left others ({} then {})",
        before.len(),
        after.len()
    );
    // A df that dropped with the tombstones would raise idf and move every
    // survivor's score.
    for row in &after {
        assert!(
            before.contains(row),
            "surviving row {row:?} changed score after the delete"
        );
    }
}

/// An update is a delete plus an insert, so until compaction both the
/// superseded row and its replacement count in df. The maintained table
/// must still equal the control, and must differ from a table built fresh
/// with the post-update content, which holds only the replacements.
#[test]
fn update_counts_superseded_and_replacement_rows() {
    let (_dir, st, _ctrl_dir, control) = maintained_and_control();

    // Rewrite segment 0's `alpha` rows to `gamma`: afterwards `alpha`
    // survives only in later segments and `gamma` only in the new rows.
    let targets = alpha_titles(0);
    let new_rows = title_batch(&gamma_rewrites(&targets));
    let updated = st
        .update(in_list_predicate(&targets), &new_rows)
        .expect("update");
    assert_eq!(updated.matched(), targets.len());
    control
        .update(in_list_predicate(&targets), &new_rows)
        .expect("update on control");

    for (query, mode) in [
        ("alpha", BoolMode::Or),
        ("gamma", BoolMode::Or),
        ("shared", BoolMode::Or),
        ("gamma shared", BoolMode::And),
    ] {
        assert_eq!(
            global_hits(&st, query, mode),
            global_hits(&control, query, mode),
            "ranking must equal the control for {query:?} after an update"
        );
    }

    let target_set: HashSet<&String> = targets.iter().collect();
    for (title, _) in global_hits(&st, "alpha shared", BoolMode::Or) {
        assert!(
            !target_set.contains(&title),
            "superseded row {title:?} resurfaced after the update"
        );
    }

    let fresh_dir = TempDir::new().expect("tempdir");
    let fresh = Supertable::create(options_with_storage(&fresh_dir)).expect("create fresh");
    commit_titles(&fresh, &gamma_rewrites(&segment_titles(0)));
    for segment in 1..SEGMENTS {
        commit_titles(&fresh, &segment_titles(segment));
    }
    let scores = |table: &Supertable| -> Vec<f32> {
        global_hits(table, "gamma", BoolMode::Or)
            .into_iter()
            .map(|(_, s)| s)
            .collect()
    };
    let (mutated, rebuilt) = (scores(&st), scores(&fresh));
    assert!(!mutated.is_empty(), "fixture must return replacement rows");
    assert_eq!(
        mutated.len(),
        rebuilt.len(),
        "both tables return the same `gamma` rows, so only scores differ"
    );
    assert_ne!(
        mutated, rebuilt,
        "an updated table scored like a fresh rebuild: the superseded rows' \
         statistics were dropped before compaction"
    );
}
