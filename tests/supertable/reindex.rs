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

use std::{sync::Arc, time::Duration};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use infino::{ReindexOptions, Supertable, superfile::format::fts::VERSION_CURRENT};

use crate::corpus_shapes::{
    N_DOCS, assert_scores_equivalent, blob_versions, corpus_dir, hits, hits_k, open_corpus,
    scores_by_id,
};

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
    let ranking_before = scores_by_id(&table, "body", "common shared", N_DOCS);
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

    // And nothing a caller can see has moved: the same documents match,
    // each with the same score. Compared by id rather than by rank, since
    // the migration reshapes the files that decide tie order.
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS),
        &ranking_before,
        shape,
    );

    assert_eq!(
        hits(&table, "body", "common"),
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

/// A table holding both migrated and unmigrated superfiles opens and
/// ranks as if it held neither kind.
///
/// This is the state a reindex passes through on every table with more
/// than one superfile: jobs commit one at a time, so between any two of
/// them the table is part old and part new. It is also the state a table
/// sits in indefinitely if a run is interrupted, and the one an append
/// creates the moment it lands beside files an older engine wrote.
///
/// Two things are mixed at once and they mix independently. The **blob
/// version** differs, so the reader decodes two layouts and corrects two
/// bound scales in one query. The **analysis revision** differs, so the
/// corpus statistics a score is normalised by fold over superfiles that
/// did not tokenize alike — the one this engine appended holds the
/// corrected terms, the ones it inherited do not.
///
/// Reached without threads or timing: append to a corpus table, which
/// puts a current superfile beside inherited ones, then reindex, which
/// leaves the revisions exactly where they were. If a mixed table
/// mis-scored, a reindex would be unsafe to interrupt and unsafe to run
/// on a table that is still taking writes — both of which it claims to
/// be.
fn assert_mixed_table_reads_cleanly(shape: &str, from_version: u32) {
    let Some((_tmp, table, root)) = open_corpus(shape) else {
        return;
    };
    let inherited = blob_versions(&root).len();
    assert!(
        inherited > 1,
        "{shape}: a single-superfile table cannot be mixed, so it proves nothing here"
    );

    // Rows whose terms this engine analyzed, landing beside rows analyzed
    // by the writer the corpus was generated with.
    let appended = ["common shared fresh", "common fresh"];
    let schema = table.schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        schema
            .fields()
            .iter()
            .map(|f| -> ArrayRef {
                let values = appended
                    .iter()
                    .map(|_| Some(f.name().as_str()))
                    .collect::<Vec<_>>();
                match f.name().as_str() {
                    "body" => Arc::new(LargeStringArray::from(appended.to_vec())),
                    _ => Arc::new(LargeStringArray::from(values)),
                }
            })
            .collect::<Vec<_>>(),
    )
    .expect("batch matches the corpus schema");
    table.append(&batch).expect("append beside inherited files");

    // Mixed on both axes now: the appended superfile is current, the
    // inherited ones are not.
    let mixed = blob_versions(&root);
    assert!(
        mixed.contains(&from_version) && mixed.contains(&VERSION_CURRENT),
        "{shape}: expected both versions present, got {mixed:?}"
    );

    // Every document still carries the corpus-wide term, inherited and
    // appended alike — so the fold over two tokenizations did not lose a
    // posting list or double-count one.
    let with_appended = N_DOCS + appended.len();
    assert_eq!(
        hits_k(&table, "body", "common", with_appended),
        with_appended,
        "{shape}: the corpus-wide term does not span both kinds of superfile"
    );
    let ranking_mixed = scores_by_id(&table, "body", "common shared", with_appended);
    assert!(
        !ranking_mixed.is_empty(),
        "{shape}: a mixed table returned nothing"
    );

    // And the rewrite leaves what a caller sees untouched, from the mixed
    // state rather than from a uniform one. The appended file is already
    // current, so the planner must skip it rather than rewrite it.
    let report = table
        .reindex(&ReindexOptions::default())
        .expect("reindex a mixed table");
    assert_eq!(
        report.already_current, 1,
        "{shape}: the superfile this engine just wrote was not recognised as current"
    );
    assert_eq!(
        report.rewritten, inherited,
        "{shape}: every inherited superfile is rewritten, and only those"
    );
    assert_eq!(
        hits_k(&table, "body", "common", with_appended),
        with_appended,
        "{shape}: the corpus-wide term stopped spanning the table after the rewrite"
    );
    // Ids *and* scores. The document set does not change, so the corpus
    // statistics a score is normalised by do not either — any drift here
    // is the bound scale being corrected once too often or not at all,
    // which is the failure a version-gated decode exists to avoid.
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", with_appended),
        &ranking_mixed,
        shape,
    );
}

#[test]
fn a_mixed_table_reads_cleanly_across_versions_and_revisions() {
    assert_mixed_table_reads_cleanly("v2_positions_region", 2);
}

/// A table with a vector index can have its terms repaired too.
///
/// Re-analysis used to be refused here: rebuilding terms went through the
/// append path, which decodes every vector back to `f32`, and only the
/// `Fp32` rerank codec survives that round trip — a codec no public API
/// reaches. Carrying the vector subsection instead of rebuilding it
/// removes the decode, and with it the limitation.
#[test]
fn a_vector_bearing_table_can_be_rewritten_and_reanalyzed() {
    let Some((_tmp, table, root)) = open_corpus("v6_hybrid") else {
        return;
    };

    let before = table.index_staleness().expect("assess a hybrid table");
    assert!(before.superfiles > 0, "the fixture has no superfiles");
    assert_eq!(
        before.awaiting_reanalysis, before.superfiles,
        "every corpus file predates the analysis revision"
    );

    let probe: Vec<f32> = infino_probe_embedding();
    let hits_before = vector_hits(&table, &probe);
    assert!(
        !hits_before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    let report = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analysis is available to a hybrid table");
    assert_eq!(
        report.rewritten, before.superfiles,
        "re-analysis did not repair every stale superfile"
    );

    table.gc(Duration::ZERO).expect("collect superseded bytes");
    assert!(
        blob_versions(&root).iter().all(|v| *v == VERSION_CURRENT),
        "the hybrid table did not reach the current container"
    );
    assert_eq!(
        vector_hits(&table, &probe),
        hits_before,
        "re-analysis moved the vector results it carries across untouched"
    );

    // Both axes clear: the container is current and the terms were rebuilt
    // from stored text, which is what a rewrite alone could never do.
    let after = table.index_staleness().expect("assess the repaired table");
    assert_eq!(after.needing_rewrite, 0, "containers are still behind");
    assert_eq!(
        after.awaiting_reanalysis, 0,
        "terms are still from an older analysis: {after:?}"
    );
    assert!(
        after.is_current(),
        "a fully repaired table must report itself finished: {after:?}"
    );
}

/// The probe used against the corpus's planted embeddings; mirrors the
/// generators' `embedding(0)`.
fn infino_probe_embedding() -> Vec<f32> {
    const DIM: usize = 16;
    (0..DIM).map(|d| if d == 0 { 1.0 } else { 0.05 }).collect()
}

/// Ids a vector search returns, in rank order.
fn vector_hits(table: &Supertable, probe: &[f32]) -> Vec<i128> {
    let batches = table
        .vector_search("emb", probe, 16, None, None)
        .expect("vector search");
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .expect("_id is Decimal128");
        for i in 0..batch.num_rows() {
            out.push(ids.value(i));
        }
    }
    out
}

/// The assessment reports what the run then does — the same numbers, not
/// a parallel estimate of them.
///
/// That equality is the whole value of the thing. An operator uses it to
/// decide whether to rewrite committed data and how large the job is; a
/// report that drifted from the planner would be worse than no report,
/// because it would be trusted. So every count is checked against the run
/// it predicts rather than against a hand-written expectation.
fn assert_staleness_predicts_the_run(shape: &str) {
    let Some((_tmp, table, _root)) = open_corpus(shape) else {
        return;
    };
    let before = table.index_staleness().expect("assess a stale table");

    assert!(!before.is_current(), "{shape}: a corpus table is behind");
    assert_eq!(
        before.superfiles, before.needing_rewrite,
        "{shape}: every superfile in a corpus table has an older container"
    );
    assert_eq!(
        before.awaiting_reanalysis, before.superfiles,
        "{shape}: every corpus file predates the analysis revision"
    );
    assert!(
        before.bytes_to_rewrite > 0,
        "{shape}: a rewrite that moves no bytes is not a rewrite"
    );

    // Assessing changes nothing: run it twice and the second answer is the
    // first. A read-only claim is cheap to make and cheap to break.
    assert_eq!(
        table.index_staleness().expect("assess again"),
        before,
        "{shape}: assessing the table changed it"
    );

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("rewrite what the assessment described");
    assert_eq!(
        report.rewritten, before.needing_rewrite,
        "{shape}: the run rewrote a different number of files than predicted"
    );
    assert_eq!(
        report.awaiting_reanalysis, before.awaiting_reanalysis,
        "{shape}: the run and the assessment disagree on what is analysis-stale"
    );
    assert_eq!(
        report.unrepairable_columns, before.unrepairable_columns,
        "{shape}: the run and the assessment name different unrepairable columns"
    );

    // After the rewrite the container axis is clear and the analysis axis
    // is not — the two-axis split, visible without running anything.
    let after = table.index_staleness().expect("assess a rewritten table");
    assert_eq!(
        after.needing_rewrite, 0,
        "{shape}: containers are still behind after a rewrite"
    );
    assert_eq!(
        after.bytes_to_rewrite, 0,
        "{shape}: a table with nothing to rewrite reports bytes to rewrite"
    );
    assert_eq!(
        after.awaiting_reanalysis, before.awaiting_reanalysis,
        "{shape}: a rewrite cleared an analysis revision, which it cannot do"
    );
    assert!(
        !after.is_current(),
        "{shape}: a rewritten but un-reanalyzed table must not look finished"
    );

    // And re-analysis is what clears it, leaving nothing to report.
    table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analyze");
    let finished = table.index_staleness().expect("assess a migrated table");
    assert!(
        finished.is_current(),
        "{shape}: a fully migrated table still reports work: {finished:?}"
    );
    assert_eq!(
        finished.superfiles, before.superfiles,
        "{shape}: files appeared or vanished"
    );
}

#[test]
fn staleness_predicts_a_multi_superfile_run() {
    assert_staleness_predicts_the_run("v2_positions_region");
}

#[test]
fn staleness_predicts_a_positional_run() {
    assert_staleness_predicts_the_run("v5_positional");
}

/// A table this engine wrote reports nothing to do, so an operator running
/// the assessment on a healthy table is told to stop rather than given a
/// number they have to interpret.
#[test]
fn a_current_table_reports_nothing_to_do() {
    let Some((_tmp, table, _root)) = open_corpus("v6_positional") else {
        return;
    };
    table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("bring the newest published shape fully current");
    let report = table.index_staleness().expect("assess");
    assert!(
        report.is_current(),
        "a migrated table reports work: {report:?}"
    );
    assert_eq!(report.needing_rewrite, 0);
    assert_eq!(report.awaiting_reanalysis, 0);
    assert_eq!(report.bytes_to_rewrite, 0);
    assert!(report.unrepairable_columns.is_empty());
    assert!(
        report.superfiles > 0,
        "the table has superfiles to be current about"
    );
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
    let before = scores_by_id(&table, "body", "common shared", N_DOCS);

    let report = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analyze the newest published shape");
    assert!(report.rewritten > 0, "nothing was re-analyzed");
    assert_eq!(
        report.awaiting_reanalysis, 0,
        "re-analysis is what clears this axis"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS),
        &before,
        "re-analysing the newest shape",
    );

    let again = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("second re-analysis");
    assert_eq!(again.rewritten, 0, "re-analysis is not idempotent");
}
