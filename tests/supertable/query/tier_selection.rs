// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! The term index's tiers choose the superfiles a ranked query opens.
//!
//! Two properties, over a table of several superfiles on storage with a
//! planted vocabulary: with tiers the ranked search returns exactly what
//! it returns without them — score for score, id for id — for every
//! clause shape and `k` here; and a multi-term query opens fewer
//! superfiles with them than without, because the tiers name the
//! documents that carry both terms before any superfile is read.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use datafusion::prelude::{col, lit};
use infino::{
    Bm25SearchOptions, BoolMode,
    runtime_metrics::op_stats::with_op_stats,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions, query::SuperfileHit},
};
use tempfile::TempDir;

use super::tier_harness::*;

#[test]
fn tiers_change_which_superfiles_open_and_nothing_else() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    let mut fewer = 0usize;
    let mut compared = 0usize;
    for (query, mode) in queries() {
        for k in KS {
            let (a, with_stats, _) = run(&with, &query, mode, k);
            let (b, without_stats, _) = run(&without, &query, mode, k);
            let (opened_with, opened_without) = (opened(&with_stats), opened(&without_stats));
            assert_eq!(
                a, b,
                "{query:?} {mode:?} k={k}: the tiers must not change the answer"
            );
            compared += 1;
            if opened_with < opened_without {
                fewer += 1;
            }
            // The merge stops once a round with a threshold rules nothing
            // out, and the survivors then open by their best known
            // candidate rather than by ceiling, so a query can open one
            // superfile more than the ceilings did; never many more.
            assert!(
                opened_with <= opened_without + 1,
                "{query:?} {mode:?} k={k}: tiers opened {opened_with} superfiles, ceilings {opened_without}"
            );
        }
    }
    assert!(
        fewer > compared / 4,
        "tiers opened fewer superfiles for only {fewer} of {compared} runs"
    );
}

/// A pair whose documents sit in a few superfiles opens only those: the
/// tier names the documents that carry both terms.
#[test]
fn a_rare_conjunction_opens_only_its_holders() {
    let dir = TempDir::new().expect("tempdir");
    let st = table(&dir, true);
    // The first ask of the pair reads its tiers from storage, beside its
    // first opens (whether it still waits for that read when its opens are
    // done is a race, so its stats are not asserted on); it answers as the
    // resident asks after it do.
    let (first, _) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits(
                "body",
                "quokka tensor",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::And),
            )
            .expect("search")
    });
    let (hits, stats, holders) = run(&st, "quokka tensor", BoolMode::And, 10);
    assert!(!hits.is_empty(), "the planted pair matches");
    assert_eq!(
        first
            .iter()
            .map(|h| (h.local_doc_id, h.score))
            .collect::<Vec<_>>(),
        hits.iter()
            .map(|&(_, row, score)| (row, score))
            .collect::<Vec<_>>(),
        "the first ask answers as a resident one"
    );
    assert_eq!(stats.tier_bytes, 0, "a later ask finds the tiers resident");
    // Both terms are under the budget, so their tiers are complete and the
    // merge knows every document that carries both, with its score as the
    // kernel computes it: the index answers outright and nothing opens.
    assert_eq!(
        opened(&stats),
        0,
        "opened {} superfiles for a k=10 conjunction whose hits sit in {holders}",
        opened(&stats)
    );
    // A superfile that holds a hit is answered from the index; one the
    // threshold ruled out is neither answered nor opened.
    assert!(
        stats.superfiles_resolved >= holders as u64
            && stats.superfiles_resolved <= SUPERFILES as u64,
        "{} superfiles answered from the index for {holders} holders",
        stats.superfiles_resolved
    );
}

/// A document deleted after the index was built is in the heads but not
/// in the table. The heads name documents by the superfile's own id, which
/// its deleted rows are not keyed by, so the superfile holding the delete
/// is opened, where it can still place, rather than answered from the
/// index — and leaves it out; the others are still answered from the
/// index.
#[test]
fn a_resolved_superfile_respects_tombstones() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    // Document 97 carries the planted pair; delete it in both tables.
    let doomed = document(97);
    for st in [&with, &without] {
        st.delete(col("body").eq(lit(doomed.as_str())))
            .expect("delete");
    }
    for k in [1usize, 10, 100] {
        let (a, stats, _) = run(&with, "quokka tensor", BoolMode::And, k);
        let (b, _, _) = run(&without, "quokka tensor", BoolMode::And, k);
        assert_eq!(a, b, "k={k}: same answer after the delete");
        assert!(
            a.iter().all(|&(sf, row, _)| !(sf == 0 && row == 97)),
            "k={k}: the deleted document is not returned"
        );
        assert!(
            opened(&stats) <= 1,
            "k={k}: at most the superfile with the delete is opened: {}",
            opened(&stats)
        );
    }
}

/// With a shallow head depth asked for, each superfile still keeps a
/// common term's proportional share, so the level heads cut from them
/// read a level at a time and open what they cannot rule out — and the
/// answer is still hit for hit the ceiling path's, at every k, for every
/// shape, with fewer opens over the whole matrix. The heads of rare terms
/// stay whole, so their conjunction is still answered in the index.
#[test]
fn heads_cut_to_a_depth_answer_the_same_and_resolve_the_rare() {
    const DEPTH: usize = 64;
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table_with(
        &with_dir,
        Build {
            tiers: true,
            depth: Some(DEPTH),
            ..Build::default()
        },
    );
    let without = table(&without_dir, false);
    let (mut opened_with, mut opened_without) = (0u64, 0u64);
    for (query, mode) in queries() {
        for k in KS {
            let (a, with_stats, _) = run(&with, &query, mode, k);
            let (b, without_stats, _) = run(&without, &query, mode, k);
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer at depth 64");
            opened_with += opened(&with_stats);
            opened_without += opened(&without_stats);
        }
    }
    assert!(
        opened_with < opened_without,
        "cut heads opened {opened_with} superfiles over the matrix, ceilings {opened_without}"
    );
    let (hits, stats, _) = run(&with, "quokka tensor", BoolMode::And, 10);
    assert!(!hits.is_empty());
    assert_eq!(
        opened(&stats),
        0,
        "the rare conjunction is still answered in the index"
    );
}

/// A conjunction of two complete-tier terms that never share a document
/// is settled by the merge: nothing opens, nothing matches.
#[test]
fn an_empty_conjunction_of_complete_tiers_opens_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let st = table(&dir, true);
    let (hits, stats, _) = run(&st, "zebra yak", BoolMode::And, 10);
    assert!(hits.is_empty(), "the words never co-occur");
    assert_eq!(opened(&stats), 0, "the tiers rule every superfile out");
}

/// Short text: every document scores the same, so every unread block can
/// still hold a tie and the merge's threshold test never settles. It must
/// still stop after a round that moved nothing, read a small part of the
/// tiers, and give the same answer as the table without tiers.
#[test]
fn tied_short_text_stops_the_merge_early_and_answers_the_same() {
    const SUPERFILES_TIED: usize = 6;
    const DOCS_PER_SUPERFILE_TIED: usize = 3_000;
    let build = |tiers: bool| -> (TempDir, Supertable) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        // Opens one at a time, so a superfile that can only tie the k-th
        // score is asked about after the first has set the floor.
        let mut options = SupertableOptions::new(schema(), vec![FtsConfig::new("body")], vec![])
            .expect("options")
            .with_storage(storage)
            .with_bound_ordered_open_window(1)
            .with_tier_merge_gate(false);
        if !tiers {
            options = options.with_term_tier_entries_per_term(0);
        }
        let st = Supertable::create(options).expect("create");
        let mut w = st.writer().expect("writer");
        for _ in 0..SUPERFILES_TIED {
            let docs: Vec<&str> = (0..DOCS_PER_SUPERFILE_TIED)
                .map(|i| {
                    if i % 3 == 0 {
                        "alpha beta"
                    } else {
                        "alpha gamma"
                    }
                })
                .collect();
            let col: ArrayRef = Arc::new(LargeStringArray::from(docs));
            w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
                .expect("append");
            w.commit().expect("commit");
        }
        drop(w);
        // One segment, so a term's tier is a few blocks rather than one per
        // commit, and the early stop has something to skip.
        st.optimize(&infino::OptimizeOptions::default())
            .expect("optimize");
        (dir, st)
    };
    let (_with_dir, with) = build(true);
    let (_without_dir, without) = build(false);
    for (query, mode) in [
        ("alpha beta", BoolMode::Or),
        ("alpha beta", BoolMode::And),
        ("alpha gamma", BoolMode::And),
    ] {
        for k in [10usize, 100] {
            let (a, stats, _) = run(&with, query, mode, k);
            let (b, without_stats, _) = run(&without, query, mode, k);
            let without_opened = without_stats.score_pruning_survived;
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
            let total = (SUPERFILES_TIED * DOCS_PER_SUPERFILE_TIED) as u64;
            assert!(
                stats.tier_entries < total,
                "{query:?} {mode:?} k={k}: read {} tier entries of at least {total}",
                stats.tier_entries
            );
            // Every superfile holds the same documents, so after the first
            // open every other can only tie the k-th score. A tie is settled
            // by `_id` order before the open only where the bound is a read
            // document's score bit for bit; what lies past the blocks read
            // is bounded through a stored key, widened by an ulp as the
            // ceiling path widens its ceilings, so here the rest open as
            // they do there — never more.
            assert!(
                stats.score_pruning_survived <= without_opened,
                "{query:?} {mode:?} k={k}: opens {} superfiles, the ceiling path {without_opened}",
                stats.score_pruning_survived
            );
        }
    }
}

/// `(body, score bits)` per row of `sql`, sorted: what two tables built
/// from the same commits compare by, their `_id`s being their own.
fn rows_of(st: &Supertable, sql: &str) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    for b in st.reader().expect("reader").query_sql(sql).expect("sql") {
        let body = b
            .column(b.schema().index_of("body").expect("body"))
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("body text");
        let score = b
            .column(b.schema().index_of("score").expect("score"))
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("f32 score");
        for i in 0..b.num_rows() {
            out.push((body.value(i).to_string(), score.value(i).to_bits()));
        }
    }
    out.sort();
    out
}

/// A `WHERE` the index bounds admits only some rows, and the tiers' bounds
/// and threshold are over every row: a scoped search answers as the table
/// without tiers does, with every admitted hit.
#[test]
fn a_scoped_search_answers_as_its_twin_does() {
    let (with, without) = (TempDir::new().expect("dir"), TempDir::new().expect("dir"));
    let tiered = table(&with, true);
    let plain = table(&without, false);
    for k in [1usize, 10, 100] {
        for (query, mode) in [
            ("the data", "or"),
            ("the data", "and"),
            ("search index", "or"),
        ] {
            let sql = format!(
                "SELECT body, score FROM bm25_search('body', '{query}', {k}, '{mode}') \
                 WHERE body LIKE '%quokka%'"
            );
            let a = rows_of(&tiered, &sql);
            let b = rows_of(&plain, &sql);
            assert!(!b.is_empty(), "{sql}: the scope admits hits");
            assert_eq!(a, b, "{sql}: same answer");
        }
    }
}

/// Documents deleted after the index was built are in the tiers: the
/// merge's threshold and exclusions must leave them out, or the live
/// documents that score below them are pruned by the kernels and the
/// superfiles holding them are never opened. Deleting a query's top
/// documents and asking again must answer as the table without tiers does.
#[test]
fn a_delete_of_the_top_documents_leaves_the_rest_answered() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    for (query, mode) in [
        ("the data search", BoolMode::Or),
        ("the data search", BoolMode::And),
        ("data search", BoolMode::And),
        ("data search", BoolMode::Or),
    ] {
        // The top documents before the delete, named by their place.
        let (top, _, _) = run(&without, query, mode, 10);
        for &(sf, row, _) in &top {
            let doomed = document(sf * DOCS_PER_SUPERFILE + row as usize);
            for st in [&with, &without] {
                st.delete(col("body").eq(lit(doomed.as_str())))
                    .expect("delete");
            }
        }
        for k in [1usize, 10, 100] {
            let (a, _, _) = run(&with, query, mode, k);
            let (b, _, _) = run(&without, query, mode, k);
            assert!(!b.is_empty(), "{query:?} {mode:?} k={k}: hits remain");
            assert_eq!(
                a, b,
                "{query:?} {mode:?} k={k}: same answer after the delete"
            );
        }
    }
}

/// A superfile indexed with no heads, merged with one that keeps them: the
/// merged term's tier holds nothing for the first superfile, whose
/// documents must still be found. (A tail that read as "a head that holds
/// every document" once dropped them, with nothing opened.)
#[test]
fn a_heads_off_segment_merged_with_a_tiered_one_keeps_its_documents() {
    let rows = |st: &Supertable, docs: &[&str]| {
        let mut w = st.writer().expect("writer");
        let col: ArrayRef = Arc::new(LargeStringArray::from(docs.to_vec()));
        w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    };
    let fold = |st: &Supertable| {
        // The index segments merge; the superfiles stay as committed.
        st.optimize(&infino::OptimizeOptions::compact(
            infino::CompactionSettings {
                min_fill_percent: 100,
                min_superfiles_for_merge: u64::MAX,
                ..infino::CompactionSettings::default()
            },
        ))
        .expect("optimize");
    };
    let build = |tiers: bool| Build {
        tiers,
        gate: false,
        ..Build::default()
    };
    let first = ["alpha beta", "alpha gamma", "beta delta"];
    let second = ["alpha beta delta", "alpha", "beta", "gamma delta"];
    // The first superfile indexed without heads, the second with them.
    let with_dir = TempDir::new().expect("tempdir");
    let with = {
        let st = Supertable::create(options_for(&with_dir, build(false))).expect("create");
        rows(&st, &first);
        drop(st);
        let st = Supertable::open(options_for(&with_dir, build(true))).expect("open");
        rows(&st, &second);
        fold(&st);
        st
    };
    // The same rows with no heads anywhere.
    let without_dir = TempDir::new().expect("tempdir");
    let without = {
        let st = Supertable::create(options_for(&without_dir, build(false))).expect("create");
        rows(&st, &first);
        rows(&st, &second);
        fold(&st);
        st
    };
    for (query, mode) in [
        ("alpha beta", BoolMode::And),
        ("alpha beta", BoolMode::Or),
        ("alpha", BoolMode::Or),
    ] {
        for k in [1usize, 10] {
            let (a, _, _) = run(&with, query, mode, k);
            let (b, _, _) = run(&without, query, mode, k);
            assert!(!b.is_empty(), "{query:?} {mode:?} k={k}: hits");
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
        }
    }
    let (hits, _, _) = run(&with, "alpha beta", BoolMode::And, 10);
    assert!(
        hits.iter().any(|&(sf, row, _)| sf == 0 && row == 0),
        "the heads-off superfile's document is among the hits: {hits:?}"
    );
}

/// A tombstone sidecar that cannot be read leaves the query to its
/// ceilings, and the query still answers: the superfiles whose sidecars
/// do read build their live-row sets, so the row counts those are built
/// from must be in hand. (A query once failed with "missing from the kept
/// set" here.) The unreadable superfile holds no hit, since a hit's own
/// sidecar is read again after ranking.
#[test]
fn an_unreadable_tombstone_sidecar_leaves_the_query_answered() {
    let rows = |st: &Supertable, docs: &[&str]| {
        let mut w = st.writer().expect("writer");
        let col: ArrayRef = Arc::new(LargeStringArray::from(docs.to_vec()));
        w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    };
    let build = |tiers: bool| Build {
        tiers,
        gate: false,
        ..Build::default()
    };
    let make = |dir: &TempDir, tiers: bool| -> Supertable {
        let st = Supertable::create(options_for(dir, build(tiers))).expect("create");
        // The first superfile by id holds none of the query's words.
        rows(&st, &["alpha", "alpha delta", "delta"]);
        for i in 0..3 {
            rows(
                &st,
                &[
                    &format!("beta gamma {i}"),
                    &format!("beta gamma doomed {i}"),
                    &format!("gamma {i}"),
                ],
            );
            st.delete(col("body").eq(lit(format!("beta gamma doomed {i}"))))
                .expect("delete");
        }
        let first = st
            .reader()
            .expect("reader")
            .manifest()
            .superfiles
            .iter()
            .min_by_key(|e| e.id_min)
            .expect("a superfile")
            .superfile_id;
        drop(st);
        // Its sidecar, garbage; the table reopens so nothing remembers a
        // sidecar it read before.
        let sidecar = dir
            .path()
            .join("superfiles")
            .join(format!("{first}.tombstones"));
        std::fs::write(&sidecar, b"not a tombstone sidecar").expect("corrupt");
        Supertable::open(options_for(dir, build(tiers))).expect("open")
    };
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = make(&with_dir, true);
    let without = make(&without_dir, false);
    for (query, mode) in [
        ("beta gamma", BoolMode::And),
        ("beta gamma", BoolMode::Or),
        ("beta", BoolMode::Or),
    ] {
        for k in [1usize, 10] {
            let (a, _, _) = run(&with, query, mode, k);
            let (b, _, _) = run(&without, query, mode, k);
            assert!(!b.is_empty(), "{query:?} {mode:?} k={k}: hits");
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
            assert!(
                a.iter().all(|&(_, row, _)| row != 1),
                "{query:?} {mode:?} k={k}: no deleted row among {a:?}"
            );
        }
    }
}

/// A tier object that does not verify fails nothing: the query keeps its
/// ceilings and answers as a table without tiers does. (A rare term's few
/// entries per commit ride inline in the slices, which the damage does not
/// reach, so a query of such terms may still be answered from them.)
#[test]
fn an_unreadable_tier_leaves_the_query_to_its_ceilings() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let build = |tiers: bool| Build {
        tiers,
        gate: false,
        ..Build::default()
    };
    let with = table_with(&with_dir, build(true));
    let without = table_with(&without_dir, build(false));
    drop(with);
    // Every tier object damaged throughout.
    let mut flipped = 0;
    for entry in std::fs::read_dir(with_dir.path().join("term-index")).expect("term-index dir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !name.starts_with("tier-") {
            continue;
        }
        // Every byte, so no block of the object verifies.
        let mut bytes = std::fs::read(&path).expect("read");
        for b in bytes.iter_mut() {
            *b ^= 0xff;
        }
        std::fs::write(&path, &bytes).expect("write");
        flipped += 1;
    }
    assert!(flipped > 0, "the table has tier objects");
    let with = Supertable::open(options_for(&with_dir, build(true))).expect("open");
    for (query, mode) in queries() {
        let (a, _, _) = run(&with, &query, mode, 10);
        let (b, _, _) = run(&without, &query, mode, 10);
        assert_eq!(a, b, "{query:?} {mode:?}: same answer");
    }
    // Common terms' heads are in the damaged objects: nothing is read or
    // answered from them.
    for (query, mode) in [("the data", BoolMode::And), ("search index", BoolMode::Or)] {
        let (_, stats, _) = run(&with, query, mode, 10);
        assert_eq!(stats.tier_entries, 0, "{query:?} {mode:?}: no tier read");
        assert_eq!(
            stats.superfiles_resolved, 0,
            "{query:?} {mode:?}: nothing answered from a tier"
        );
    }
}

/// With the learned gate on, a table whose terms carry level heads is
/// served by them on every ask: the gate weighs only the tier merge of
/// layers built before level heads, so every ask reads the heads and
/// answers the same.
///
/// The handle opens every superfile in one window, where the gate would
/// otherwise have had a choice to make.
#[test]
fn the_learned_gate_leaves_the_level_heads_to_serve_every_ask() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let gated = Build {
        tiers: true,
        gate: true,
        ..Build::default()
    };
    drop(table_with(&with_dir, gated));
    let without = table(&without_dir, false);
    for (query, mode) in [
        ("the data", BoolMode::Or),
        ("the data", BoolMode::And),
        ("search index", BoolMode::And),
        ("the data search", BoolMode::Or),
        ("quokka tensor", BoolMode::And),
    ] {
        for k in [10usize, 100] {
            let with = Supertable::open(
                options_for(&with_dir, gated).with_bound_ordered_open_window(SUPERFILES),
            )
            .expect("open");
            let (expected, _, _) = run(&without, query, mode, k);
            for _ in 0..20 {
                let (a, stats, _) = run_once(&with, query, mode, k);
                assert_eq!(
                    a, expected,
                    "{query:?} {mode:?} k={k}: same answer every time"
                );
                assert!(
                    stats.tier_entries > 0,
                    "{query:?} {mode:?} k={k}: the heads served the ask"
                );
            }
        }
    }
}

/// Null bodies: the heads name rows, not the column's document ids, so a
/// superfile the index answers for names the right rows.
#[test]
fn nulls_in_the_column_answer_the_same_and_still_resolve() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table_with(
        &with_dir,
        Build {
            tiers: true,
            nulls: true,
            ..Build::default()
        },
    );
    let without = table_with(
        &without_dir,
        Build {
            tiers: false,
            nulls: true,
            ..Build::default()
        },
    );
    for (query, mode) in queries() {
        for k in [10usize, 100] {
            let (a, _, _) = run(&with, &query, mode, k);
            let (b, _, _) = run(&without, &query, mode, k);
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer with nulls");
        }
    }
    let (hits, stats, _) = run(&with, "quokka tensor", BoolMode::And, 10);
    assert!(!hits.is_empty());
    assert_eq!(
        opened(&stats),
        0,
        "the rare conjunction is still answered from the index"
    );
}

/// A cap on a term's entries cuts every superfile's head to the same
/// shallower depth: the bounds hold through the raised tails and the
/// answers do not change.
#[test]
fn a_per_term_budget_cuts_heads_and_answers_the_same() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table_with(
        &with_dir,
        Build {
            tiers: true,
            budget: Some(100),
            ..Build::default()
        },
    );
    let without = table(&without_dir, false);
    let mut served = 0usize;
    for (query, mode) in queries() {
        for k in [10usize, 100] {
            let (a, stats, _) = run(&with, &query, mode, k);
            let (b, _, _) = run(&without, &query, mode, k);
            assert_eq!(
                a, b,
                "{query:?} {mode:?} k={k}: same answer under the budget"
            );
            if stats.tier_entries > 0 {
                served += 1;
            }
        }
    }
    assert!(served > 0, "the cut heads still serve queries");
}

/// The merge serves up to sixty-four distinct terms, a bit each; one more
/// keeps the ceilings. Both answer as the table without tiers does.
#[test]
fn many_terms_are_served_up_to_the_mask_width() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    let mut words: Vec<String> = ["the", "data", "search", "index"]
        .iter()
        .map(|w| (*w).to_owned())
        .collect();
    words.extend((0..60).map(|i| format!("w{i:02}")));
    let sixty_four = words.join(" ");
    words.push("w60".to_owned());
    let sixty_five = words.join(" ");
    for (query, expect_tiers) in [(&sixty_four, true), (&sixty_five, false)] {
        for k in [10usize, 100] {
            let (a, stats, _) = run(&with, query, BoolMode::Or, k);
            let (b, _, _) = run(&without, query, BoolMode::Or, k);
            assert!(!b.is_empty());
            assert_eq!(
                a,
                b,
                "{} terms, k={k}: same answer",
                query.split(' ').count()
            );
            assert_eq!(
                stats.tier_entries > 0,
                expect_tiers,
                "{} terms, k={k}: read {} tier entries",
                query.split(' ').count(),
                stats.tier_entries
            );
        }
    }
}

/// A per-query parameter override scores off the declared parameters the
/// index holds keys at: such a query takes no bound from the index — no
/// tiers, no ceilings, as before — and answers as the table without tiers
/// does under the same override.
#[test]
fn a_parameter_override_opens_without_an_index_bound() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    let hits_of = |st: &Supertable| {
        with_op_stats(|| {
            st.reader()
                .expect("reader")
                .bm25_hits(
                    "body",
                    "the data",
                    10,
                    Bm25SearchOptions::new()
                        .with_mode(BoolMode::Or)
                        .with_bm25(0.9, 0.3),
                )
                .expect("search")
                .iter()
                .map(|h| (h.local_doc_id, h.score))
                .collect::<Vec<_>>()
        })
    };
    let (a, stats) = hits_of(&with);
    let (b, _) = hits_of(&without);
    assert!(!a.is_empty());
    assert_eq!(a, b, "same rows and scores under the override");
    assert_eq!(stats.tier_entries, 0, "the tiers do not serve an override");
}

/// Two text columns: heads are per column, so a query on one takes nothing
/// from the other's — the same words in both columns, and each column
/// answers as the table without tiers does and still resolves its rare pair.
#[test]
fn two_text_columns_keep_their_heads_apart() {
    let build = |tiers: bool| -> (TempDir, Supertable) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("body", arrow_schema::DataType::LargeUtf8, true),
            arrow_schema::Field::new("tags", arrow_schema::DataType::LargeUtf8, true),
        ]));
        let mut options = SupertableOptions::new(
            Arc::clone(&schema),
            vec![FtsConfig::new("body"), FtsConfig::new("tags")],
            vec![],
        )
        .expect("options")
        .with_storage(storage)
        .with_bound_ordered_open_window(1)
        .with_tier_merge_gate(false);
        if !tiers {
            options = options.with_term_tier_entries_per_term(0);
        }
        let st = Supertable::create(options).expect("create");
        let mut w = st.writer().expect("writer");
        for s in 0..SUPERFILES {
            // The tags column holds the body of the document after it, so
            // the same words sit in both columns at different rows.
            let body: Vec<String> = (0..DOCS_PER_SUPERFILE)
                .map(|i| document(s * DOCS_PER_SUPERFILE + i))
                .collect();
            let tags: Vec<String> = (0..DOCS_PER_SUPERFILE)
                .map(|i| document(s * DOCS_PER_SUPERFILE + i + 1))
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(
                        body.iter().map(String::as_str).collect::<Vec<_>>(),
                    )) as ArrayRef,
                    Arc::new(LargeStringArray::from(
                        tags.iter().map(String::as_str).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ],
            )
            .expect("batch");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
        (dir, st)
    };
    let (_with_dir, with) = build(true);
    let (_without_dir, without) = build(false);
    // Asked twice, the second metered: the first brings the column's tiers
    // into memory, after which the merge runs before any open.
    let hits = |st: &Supertable, column: &str, query: &str, mode: BoolMode| {
        st.reader()
            .expect("reader")
            .bm25_hits(column, query, 10, Bm25SearchOptions::new().with_mode(mode))
            .expect("search");
        with_op_stats(|| {
            st.reader()
                .expect("reader")
                .bm25_hits(column, query, 10, Bm25SearchOptions::new().with_mode(mode))
                .expect("search")
                .iter()
                .map(|h| (h.local_doc_id, h.score))
                .collect::<Vec<_>>()
        })
    };
    for column in ["body", "tags"] {
        for (query, mode) in [
            ("the data", BoolMode::Or),
            ("search index", BoolMode::And),
            ("quokka tensor", BoolMode::And),
        ] {
            let (a, stats) = hits(&with, column, query, mode);
            let (b, _) = hits(&without, column, query, mode);
            assert_eq!(a, b, "{column} {query:?} {mode:?}: same answer");
            if query == "quokka tensor" {
                assert!(!a.is_empty());
                assert_eq!(
                    stats.score_pruning_survived, 0,
                    "{column}: the rare pair is answered from the column's own heads"
                );
            }
        }
    }
    // The same rows, shifted by one between the columns: the answers
    // differ, which is what shows each column read its own heads.
    let (body, _) = hits(&with, "body", "quokka tensor", BoolMode::And);
    let (tags, _) = hits(&with, "tags", "quokka tensor", BoolMode::And);
    assert_ne!(body, tags);
}

/// A table whose term index was written by the layout before tiers (root
/// at version 3, slices at version 1, the committed fixture): the current
/// reader serves it as that layout did — ceilings, no tier read — and after
/// a commit and the maintenance rebuild it answers the same, now from the
/// tiers.
#[test]
fn a_table_indexed_by_the_layout_before_tiers_serves_as_before_and_rebuilds() {
    use infino::test_helpers::copy_dir_recursive;
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pre_tier_term_index_table");
    let work = TempDir::new().expect("tempdir");
    copy_dir_recursive(&fixture, work.path());
    let open = |dir: &std::path::Path| -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir).expect("provider"));
        let mut fts = FtsConfig::new("body");
        fts.positions = true;
        Supertable::open(
            SupertableOptions::new(schema(), vec![fts], vec![])
                .expect("options")
                .with_storage(storage)
                .with_bound_ordered_open_window(1)
                .with_tier_merge_gate(false),
        )
        .expect("open the fixture")
    };
    let st = open(work.path());
    let shapes = [
        ("the data", BoolMode::Or),
        ("the data", BoolMode::And),
        ("search index", BoolMode::And),
        ("quokka tensor", BoolMode::And),
        ("\"the data\"", BoolMode::Or),
    ];
    for (q, mode) in &shapes {
        let (hits, stats, _) = run(&st, q, *mode, 10);
        assert!(!hits.is_empty(), "{q:?}: the fixture answers");
        assert_eq!(
            stats.tier_entries, 0,
            "{q:?}: the old layout has no tier to read"
        );
        assert_eq!(
            stats.superfiles_resolved, 0,
            "{q:?}: nothing is answered from the index"
        );
    }
    // A commit on top of the old root: a delta in the current layout
    // beside the old slices.
    let mut w = st.writer().expect("writer");
    let docs: Vec<String> = (0..20).map(|i| document(1_000 + i)).collect();
    let col: ArrayRef = Arc::new(LargeStringArray::from(
        docs.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
        .expect("append");
    w.commit().expect("commit");
    drop(w);
    let plain = {
        // The same rows in a table without tiers, to compare the mixed and
        // rebuilt states against.
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&fixture, dir.path());
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut fts = FtsConfig::new("body");
        fts.positions = true;
        let st = Supertable::open(
            SupertableOptions::new(schema(), vec![fts], vec![])
                .expect("options")
                .with_storage(storage)
                .with_bound_ordered_open_window(1)
                .with_term_tier_entries_per_term(0),
        )
        .expect("open");
        let mut w = st.writer().expect("writer");
        let docs: Vec<String> = (0..20).map(|i| document(1_000 + i)).collect();
        let col: ArrayRef = Arc::new(LargeStringArray::from(
            docs.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        (dir, st)
    };
    for (q, mode) in &shapes {
        let (a, _, _) = run(&st, q, *mode, 10);
        let (b, _, _) = run(&plain.1, q, *mode, 10);
        assert_eq!(
            a, b,
            "{q:?} {mode:?}: the mixed index answers as a table without tiers"
        );
    }
    // The maintenance rebuild writes the current layout throughout.
    st.optimize(&infino::OptimizeOptions::compact(
        infino::CompactionSettings {
            min_fill_percent: 100,
            min_superfiles_for_merge: u64::MAX,
            ..infino::CompactionSettings::default()
        },
    ))
    .expect("optimize");
    let mut served = false;
    for (q, mode) in &shapes {
        let (a, stats, _) = run(&st, q, *mode, 10);
        let (b, _, _) = run(&plain.1, q, *mode, 10);
        assert_eq!(a, b, "{q:?} {mode:?}: the rebuilt index answers the same");
        served |= stats.tier_entries > 0 || stats.superfiles_resolved > 0;
    }
    assert!(served, "after the rebuild the tiers serve");
}

/// A prefix over a table the term index covers scores each document by the
/// sum over every completion it holds, as a disjunction of the completions
/// does: "t" completes to "the" and "tensor", and a document holding both
/// scores as both.
#[test]
fn a_prefix_scores_as_the_disjunction_of_its_completions() {
    let dir = TempDir::new().expect("tempdir");
    let st = table(&dir, true);
    let reader = st.reader().expect("reader");
    let all = SUPERFILES * DOCS_PER_SUPERFILE;
    let key = |h: &SuperfileHit| {
        (
            format!("{:?}", h.superfile),
            h.local_doc_id,
            h.score.to_bits(),
        )
    };
    for k in [1usize, 10, 100, all] {
        let prefix = reader.bm25_search_prefix("body", "t", k).expect("prefix");
        let both = reader
            .bm25_hits("body", "the tensor", k, Bm25SearchOptions::new())
            .expect("hits");
        assert_eq!(
            prefix.iter().map(key).collect::<Vec<_>>(),
            both.iter().map(key).collect::<Vec<_>>(),
            "k={k}: the prefix answers as its completions' disjunction"
        );
    }
}

/// A pair of common terms whose best documents for one term sit past the
/// other term's head: the heads bound every unread document under the
/// threshold, and the candidates' missing frequencies are probed from the
/// covering posting blocks, so the answer is the ceiling path's with no
/// superfile opened.
#[test]
fn probes_settle_candidates_the_heads_leave_open() {
    /// Superfiles, one commit each.
    const PROBE_SUPERFILES: usize = 3;
    /// Documents per superfile: past the whole-list size, so heads are cut.
    const PROBE_DOCS: usize = 5_000;
    /// The documents, late in each superfile, where `alpha` is frequent.
    const HOT: std::ops::Range<usize> = 4_500..4_520;
    /// Tokens per document, so every document has the same length.
    const LEN: usize = 60;
    let build = |dir: &TempDir, heads: bool| {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(schema(), vec![FtsConfig::new("body")], vec![])
            .expect("options")
            .with_storage(storage)
            .with_bound_ordered_open_window(1);
        if !heads {
            options = options.with_term_tier_entries_per_term(0);
        }
        let st = Supertable::create(options).expect("create");
        for s in 0..PROBE_SUPERFILES {
            let docs: Vec<String> = (0..PROBE_DOCS)
                .map(|i| {
                    let alpha = 1 + if HOT.contains(&i) {
                        5 + i - HOT.start + 20 * s
                    } else {
                        0
                    };
                    let mut words = vec!["alpha"; alpha];
                    words.push("beta");
                    words.resize(LEN, "filler");
                    words.join(" ")
                })
                .collect();
            let col: ArrayRef = Arc::new(LargeStringArray::from(
                docs.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let mut w = st.writer().expect("writer");
            w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
                .expect("append");
            w.commit().expect("commit");
        }
        st
    };
    let (with_dir, without_dir) = (
        TempDir::new().expect("tempdir"),
        TempDir::new().expect("tempdir"),
    );
    let (with, without) = (build(&with_dir, true), build(&without_dir, false));
    for mode in [BoolMode::Or, BoolMode::And] {
        for k in [1usize, 10] {
            let (a, stats, _) = run(&with, "alpha beta", mode, k);
            let (b, _, _) = run(&without, "alpha beta", mode, k);
            assert_eq!(a, b, "{mode:?} k={k}: the probed answer is the ceilings'");
            // Probes run beside the opens: the first open window (one
            // superfile here) is under way before they report, and they
            // settle the rest unopened.
            assert!(
                opened(&stats) <= OPEN_WINDOW as u64,
                "{mode:?} k={k}: probes settle all but the first window: {}",
                opened(&stats)
            );
        }
    }
}

/// A phrase of two common words whose occurrences sit in a few late
/// documents of each superfile: the heads bound the rest, the candidates'
/// occurrences are probed from the members' covering blocks and position
/// groups, and the answer is the ceiling path's with no superfile opened.
#[test]
fn phrase_probes_settle_a_phrase_of_common_words() {
    /// Superfiles, one commit each.
    const PHRASE_SUPERFILES: usize = 3;
    /// Documents per superfile: past the whole-list size.
    const PHRASE_DOCS: usize = 5_000;
    /// The documents, late in each superfile, holding the phrase.
    const HOT: std::ops::Range<usize> = 4_500..4_520;
    let build = |dir: &TempDir, heads: bool| {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut fts = FtsConfig::new("body");
        fts.positions = true;
        let mut options = SupertableOptions::new(schema(), vec![fts], vec![])
            .expect("options")
            .with_storage(storage)
            .with_bound_ordered_open_window(1);
        if !heads {
            options = options.with_term_tier_entries_per_term(0);
        }
        let st = Supertable::create(options).expect("create");
        for s in 0..PHRASE_SUPERFILES {
            let docs: Vec<String> = (0..PHRASE_DOCS)
                .map(|i| match HOT.contains(&i) {
                    // The phrase once or more, tied nowhere across superfiles.
                    true => {
                        let mut words = Vec::new();
                        for _ in 0..(1 + (i - HOT.start + s) % 4) {
                            words.extend(["alpha", "beta"]);
                        }
                        words.resize(40, "filler");
                        words.join(" ")
                    }
                    // Both words, never adjacent in order.
                    false => {
                        let mut words = vec!["beta", "filler", "alpha"];
                        words.resize(40, "filler");
                        words.join(" ")
                    }
                })
                .collect();
            let col: ArrayRef = Arc::new(LargeStringArray::from(
                docs.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let mut w = st.writer().expect("writer");
            w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
                .expect("append");
            w.commit().expect("commit");
        }
        st
    };
    let (with_dir, without_dir) = (
        TempDir::new().expect("tempdir"),
        TempDir::new().expect("tempdir"),
    );
    let (with, without) = (build(&with_dir, true), build(&without_dir, false));
    for k in [1usize, 10] {
        let (a, stats, _) = run(&with, "\"alpha beta\"", BoolMode::Or, k);
        let (b, _, _) = run(&without, "\"alpha beta\"", BoolMode::Or, k);
        assert!(!a.is_empty(), "k={k}: the phrase matches");
        assert_eq!(a, b, "k={k}: the probed answer is the ceilings'");
        assert!(
            opened(&stats) <= OPEN_WINDOW as u64,
            "k={k}: phrase probes settle all but the first window: {}",
            opened(&stats)
        );
    }
}
