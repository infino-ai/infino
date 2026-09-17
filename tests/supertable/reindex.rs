// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reindexing a table written by an older engine.
//!
//! The corpus tables are the input these are worth running against: real
//! bytes from real releases, not files this engine wrote and then pretended
//! were old. A rewrite has to bring every superfile to the current format
//! while leaving the rows, their ids and their ranking exactly as they
//! were — a migration that changed answers would be a worse outcome than
//! the staleness it set out to fix.

use std::time::Duration;

use infino::{
    Bm25SearchOptions, ReindexOptions, Supertable, superfile::format::fts::VERSION_CURRENT,
};

use crate::corpus_shapes::{N_DOCS, blob_versions, corpus_dir, hits, open_corpus};

/// Rows a `bm25_search` returns, as `(id, score)` pairs in rank order, so
/// a comparison sees any reordering and not merely a changed count.
fn ranked(table: &Supertable, column: &str, query: &str, k: usize) -> Vec<(i128, f32)> {
    let batches = table
        .bm25_search(column, query, k, Bm25SearchOptions::new(), None)
        .expect("bm25 search");
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("_id is Decimal128");
        let scores = batch
            .column_by_name("score")
            .expect("score column")
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("score is f32");
        for i in 0..batch.num_rows() {
            out.push((ids.value(i), scores.value(i)));
        }
    }
    out
}

/// Rewriting a table written by an older engine brings every superfile to
/// the current format and changes nothing a caller can observe.
fn assert_reindex_migrates(shape: &str, from_version: u32) {
    let Some(_) = corpus_dir(shape) else {
        return;
    };
    let (_tmp, table, root) = open_corpus(shape).expect("corpus present");

    let before = blob_versions(&root);
    assert!(
        before.iter().all(|v| *v == from_version),
        "{shape}: expected every superfile at version {from_version}, got {before:?}"
    );
    let ranking_before = ranked(&table, "body", "common shared", 64);
    assert!(!ranking_before.is_empty());

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("reindex a table written by an older engine");
    assert_eq!(
        report.rewritten,
        before.len(),
        "{shape}: every stale superfile is rewritten"
    );

    // The point of the exercise: the files this table reads are now what
    // this engine writes.
    //
    // A rewrite replaces a superfile's manifest entry; the superseded
    // bytes stay on disk until they are collected, so the directory holds
    // both until then. Collect with no safety gap first, so what is left
    // is exactly the live set and the assertion is about the table rather
    // than about the timing of a sweep.
    table.gc(Duration::ZERO).expect("collect superseded bytes");
    let after = blob_versions(&root);
    assert!(
        after.iter().all(|v| *v == VERSION_CURRENT),
        "{shape}: superfiles still at {after:?} after a reindex"
    );

    // And nothing a caller can see has moved. Ids and scores, in rank
    // order — a migration that reordered results would be worse than the
    // staleness it fixes.
    let ranking_after = ranked(&table, "body", "common shared", 64);
    assert_eq!(
        ranking_before.len(),
        ranking_after.len(),
        "{shape}: the hit count changed"
    );
    for (i, (before, after)) in ranking_before.iter().zip(&ranking_after).enumerate() {
        assert_eq!(
            before.0, after.0,
            "{shape}: rank {i} is a different document"
        );
    }

    assert_eq!(
        table
            .bm25_search("body", "common", N_DOCS, Bm25SearchOptions::new(), None)
            .expect("search")
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>(),
        N_DOCS,
        "{shape}: the corpus-wide term stopped matching every document"
    );

    // Re-running rewrites nothing. Staleness is read from the files, so a
    // finished migration is self-evident and an interrupted one resumes
    // without a journal — but the termination argument is sharper than
    // that: these files were written by releases that predate the analysis
    // revision, so they stay analysis-stale even once their container is
    // current. A planner that treated that as work to redo would rewrite
    // the whole corpus on every run and never converge. The report says so
    // instead.
    let again = table
        .reindex(&ReindexOptions::default())
        .expect("a second reindex is a no-op");
    assert_eq!(again.rewritten, 0, "{shape}: reindex is not idempotent");
    assert_eq!(
        again.awaiting_reanalysis,
        before.len(),
        "{shape}: a rewritten file still holds terms from an older analysis, \
         and the report has to say so rather than plan another rewrite"
    );
    assert_eq!(
        report.awaiting_reanalysis, again.awaiting_reanalysis,
        "{shape}: a rewrite does not change how many files need re-analysis"
    );
}

#[test]
fn migrates_a_pre_positions_table() {
    assert_reindex_migrates("v2_positions_region", 2);
}

#[test]
fn migrates_a_bitset_block_table() {
    assert_reindex_migrates("v4_bitset_blocks", 4);
}

#[test]
fn migrates_a_coarse_table() {
    assert_reindex_migrates("v5_positionless", 5);
}

#[test]
fn migrates_a_positional_table() {
    assert_reindex_migrates("v5_positional", 5);
}

/// The newest published shape, whose stored bounds are already in the
/// scorer's scale rather than carrying the `(k1 + 1)` factor older files
/// do. A rewrite must leave them alone; correcting them a second time, as
/// it must for every shape above, would shrink bounds that are already
/// exact and silently prune documents out of the top-k. The ranking
/// comparison is what catches that.
#[test]
fn migrates_a_current_scale_table() {
    assert_reindex_migrates("v6_positional", 6);
}

/// Re-analysis is the only repair that changes a file's terms, so it is
/// the only one these can be asserted against.
///
/// Both terms are planted by the corpus and unreachable in every shipped
/// file: an unbroken run past the token cap was indexed whole, so the
/// capped piece a query now looks up was never written; and emoji fell out
/// of the standard analyzer as though they were punctuation. `corpus_shapes`
/// pins them at zero hits. Finding them here is the proof that terms were
/// rebuilt rather than copied.
fn assert_reanalysis_repairs_terms(shape: &str, expect_emoji: bool) {
    let Some((_tmp, table, _root)) = open_corpus(shape) else {
        return;
    };
    let capped_piece = "z".repeat(255);

    assert_eq!(
        hits(&table, "body", &capped_piece),
        0,
        "{shape}: the over-cap run is reachable before re-analysis"
    );
    let rows_before = hits(&table, "body", "common");

    let report = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analyze a table written by an older engine");
    assert!(report.rewritten > 0, "{shape}: nothing was re-analyzed");
    assert_eq!(
        report.awaiting_reanalysis, 0,
        "{shape}: re-analysis is what clears this axis"
    );
    assert!(
        report.unrepairable_columns.is_empty(),
        "{shape}: every column here stores its text"
    );

    // The run is chopped into capped pieces now, so its leading piece is a
    // real term an exact query reaches.
    assert_eq!(
        hits(&table, "body", &capped_piece),
        1,
        "{shape}: the over-cap run is still unreachable after re-analysis"
    );

    // Emoji are a standard-analyzer term. An `ascii_lower` column drops
    // them whatever the revision, so only the tables written with
    // `standard` gain them — asserting otherwise would be asserting a bug.
    let emoji = hits(&table, "body", "🔥");
    match expect_emoji {
        true => assert_eq!(emoji, 1, "{shape}: emoji still absent after re-analysis"),
        false => assert_eq!(emoji, 0, "{shape}: ascii_lower does not index emoji"),
    }

    assert_eq!(
        hits(&table, "body", "common"),
        rows_before,
        "{shape}: re-analysis changed which documents carry the corpus-wide term"
    );

    // Idempotent for the same reason a rewrite is: the files now record
    // this engine's revision, so a second pass plans nothing.
    let again = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("second re-analysis");
    assert_eq!(again.rewritten, 0, "{shape}: re-analysis is not idempotent");
}

#[test]
fn reanalysis_repairs_an_ascii_lower_table() {
    assert_reanalysis_repairs_terms("v4_bitset_blocks", false);
}

#[test]
fn reanalysis_repairs_a_standard_analyzer_table() {
    assert_reanalysis_repairs_terms("v5_positionless", true);
}

#[test]
fn reanalysis_repairs_a_positional_table() {
    assert_reanalysis_repairs_terms("v5_positional", true);
}

/// Re-analyzing the newest published shape changes no terms — it already
/// holds the ones this engine emits — and that is the case worth pinning.
///
/// The file is planned for re-analysis because it records no revision, not
/// because anything is known to be wrong with it. So the run has to end
/// somewhere: it clears the axis by recording the revision it just
/// analyzed at, and a second run plans nothing. Were the revision written
/// from the file rather than from the work done, this would re-tokenize
/// the corpus on every run, forever.
#[test]
fn reanalysis_of_the_newest_shape_converges_without_changing_terms() {
    const SHAPE: &str = "v6_positional";
    let Some((_tmp, table, _root)) = open_corpus(SHAPE) else {
        return;
    };
    let before = ranked(&table, "body", "common shared", 64);

    let report = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analyze the newest published shape");
    assert!(report.rewritten > 0, "nothing was re-analyzed");
    assert_eq!(
        report.awaiting_reanalysis, 0,
        "re-analysis is what clears this axis"
    );
    assert_eq!(
        ranked(&table, "body", "common shared", 64),
        before,
        "re-analysis moved terms that were already current"
    );

    let again = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("second re-analysis");
    assert_eq!(again.rewritten, 0, "re-analysis is not idempotent");
}
