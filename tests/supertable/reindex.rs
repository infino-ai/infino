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

use std::{collections::HashMap, fs, sync::Arc, time::Duration};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use bytes::Bytes;
use futures::executor::block_on;
use infino::{
    ReindexMode, ReindexOptions,
    superfile::{
        SuperfileReader, VectorSearchOptions,
        format::{fts::VERSION_CURRENT, kv},
    },
    supertable::manifest::SuperfileEntry,
};

use crate::corpus_shapes::{
    N_DOCS, assert_scores_equivalent, blob_versions, corpus_dir, first_region, hits, hits_k,
    open_corpus, probe_embedding, raw_footer_kvs, scores_by_id, table_dir, vector_hits,
};

/// Neighbours a footer-only open is probed for, matching the table-level
/// probe so both see the same depth of the index.
const FOOTER_PROBE_NEIGHBOURS: usize = 16;

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
    let ranking_before = scores_by_id(&table, "body", "common shared", N_DOCS as usize);
    assert!(!ranking_before.is_empty());

    let report = table
        .reindex(&ReindexOptions::rewriting())
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
        &scores_by_id(&table, "body", "common shared", N_DOCS as usize),
        &ranking_before,
        shape,
    );

    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS as usize,
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
        .reindex(&ReindexOptions::rewriting())
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
    let with_appended = N_DOCS as usize + appended.len();
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
        .reindex(&ReindexOptions::rewriting())
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
    let Some((_tmp, table, root)) = open_corpus("v6_with_vectors") else {
        return;
    };

    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess a hybrid table");
    assert!(before.superfiles > 0, "the fixture has no superfiles");
    assert_eq!(
        before.awaiting_reanalysis, before.superfiles,
        "every corpus file predates the analysis revision"
    );

    let probe: Vec<f32> = probe_embedding();
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
    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess the repaired table");
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
    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess a stale table");

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
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess again"),
        before,
        "{shape}: assessing the table changed it"
    );

    let report = table
        .reindex(&ReindexOptions::rewriting())
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
    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess a rewritten table");
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
    let finished = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess a migrated table");
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
    let report = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
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

    // The report is the run's own account of itself. Re-reading the files
    // is what catches a run that reported a clean axis while skipping the
    // superfiles that were on it.
    assert!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess after re-analysis")
            .is_current(),
        "{shape}: the files still say they are behind"
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
    let before = scores_by_id(&table, "body", "common shared", N_DOCS as usize);

    let report = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("re-analyze the newest published shape");
    assert!(report.rewritten > 0, "nothing was re-analyzed");
    assert_eq!(
        report.awaiting_reanalysis, 0,
        "re-analysis is what clears this axis"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS as usize),
        &before,
        "re-analysing the newest shape",
    );

    let again = table
        .reindex(&ReindexOptions::reanalyzing())
        .expect("second re-analysis");
    assert_eq!(again.rewritten, 0, "re-analysis is not idempotent");
}

/// The default mode leaves nothing stale, where the cheap one cannot.
///
/// This is the difference between the two that matters to a caller: a
/// layout-only repair brings the container current and reports the terms
/// it could not fix, so the table is still behind when it finishes.
/// `Auto` repairs both axes, and repairs them in one run.
#[test]
fn the_default_mode_leaves_the_table_current_where_a_rewrite_cannot() {
    const SHAPE: &str = "v5_positional";
    let Some(_) = corpus_dir(SHAPE) else {
        return;
    };

    // The cheap mode, for contrast: containers current, terms still old.
    let (_tmp, rewritten, _root) = open_corpus(SHAPE).expect("corpus present");
    rewritten
        .reindex(&ReindexOptions::rewriting())
        .expect("rewrite");
    let after_rewrite = rewritten
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert_eq!(after_rewrite.needing_rewrite, 0);
    assert!(
        after_rewrite.awaiting_reanalysis > 0 && !after_rewrite.is_current(),
        "a layout-only repair has to leave the analysis axis behind, or \
         this test is not comparing two different things: {after_rewrite:?}"
    );

    let (_tmp2, table, _root2) = open_corpus(SHAPE).expect("corpus present");
    let report = table
        .reindex(&ReindexOptions::default())
        .expect("the default repairs a table written by an older engine");
    assert!(report.rewritten > 0, "{report:?}");
    assert_eq!(
        report.awaiting_reanalysis, 0,
        "the default left files waiting on a repair it performs: {report:?}"
    );

    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert!(
        after.is_current(),
        "the default mode finished with the table still stale: {after:?}"
    );

    // The terms the older analysis could not produce are reachable, which
    // is what a re-analysis buys over a rewrite.
    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS as usize,
        "the corpus-wide term stopped matching every document"
    );

    // And it converges: a second run has nothing left to do.
    let again = table
        .reindex(&ReindexOptions::default())
        .expect("second run");
    assert_eq!(
        (again.rewritten, again.awaiting_reanalysis),
        (0, 0),
        "{again:?}"
    );
}

/// Vector hits from a superfile opened from its bytes alone, with no
/// manifest hints, so every region is located through the footer.
fn footer_only_vector_hits(bytes: &Bytes, probe: &[f32]) -> Vec<(u32, f32)> {
    let reader = SuperfileReader::open(bytes.clone()).expect("open superfile from its footer");
    block_on(reader.vector_hits_async(
        "emb",
        probe,
        FOOTER_PROBE_NEIGHBOURS,
        VectorSearchOptions::default(),
    ))
    .expect("footer-only vector search")
}

/// A reindex that carries a superfile's vector subsection writes a footer
/// that describes the file it wrote, not the one it read.
///
/// The carry path reuses the input's body and vector bytes but rebuilds
/// the FTS blob, which changes size, so the vector bytes move. Every
/// region key has to follow them, and has to be stored exactly once: a
/// stale copy left beside the right one is invisible to this engine's
/// reader, which keeps the last value, and misleads any reader that keeps
/// the first.
#[test]
fn a_carried_vector_subsection_gets_a_footer_that_describes_it() {
    let Some((_tmp, table, root)) = open_corpus("v6_with_vectors") else {
        return;
    };
    let dir = table_dir(&root);
    let probe = probe_embedding();

    // Inputs keyed by their vector bytes: a carried subsection is copied
    // byte for byte, so that is what pairs an output with its input.
    let mut inputs: HashMap<Bytes, Bytes> = HashMap::new();
    for entry in fs::read_dir(dir.join("data")).expect("read data dir") {
        let bytes = Bytes::from(fs::read(entry.expect("dir entry").path()).expect("read input"));
        let kvs = raw_footer_kvs(&bytes);
        let (at, len) = first_region(&kvs, kv::VEC_OFFSET, kv::VEC_LENGTH)
            .expect("every corpus superfile carries a vector subsection");
        inputs.insert(bytes.slice(at as usize..(at + len) as usize), bytes);
    }
    assert!(!inputs.is_empty(), "the fixture has no superfiles");

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("reindex a hybrid table");
    assert_eq!(
        report.rewritten,
        inputs.len(),
        "every superfile is rewritten"
    );
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    let reader = table.local_handle().reader().expect("reader");
    let entries = reader.manifest().get_all_superfiles();
    assert_eq!(entries.len(), inputs.len(), "one output per input");

    for entry in entries {
        let path = entry.storage_path();
        let bytes = Bytes::from(fs::read(dir.join(&path)).expect("read output"));
        // (a) and (c), plus no region key stored twice.
        let (vec_at, vec_len) = assert_footer_describes_layout(&path, &bytes, entry);

        // (b) The bytes there are the input's vector subsection.
        let vec_bytes = bytes.slice(vec_at as usize..(vec_at + vec_len) as usize);
        let input = inputs
            .get(&vec_bytes)
            .unwrap_or_else(|| panic!("{path}: vector bytes match no input's subsection"));

        // (d) Opened from the footer alone, the output answers as its input.
        assert_eq!(
            footer_only_vector_hits(&bytes, &probe),
            footer_only_vector_hits(input, &probe),
            "{path}: a footer-only open changed the vector results"
        );
    }
}

/// Asserts that `bytes`' footer describes the file it ends: every region
/// key stored once, the vector range inside the file and right after the
/// FTS blob, and both ranges agreeing with the manifest `entry`. Returns
/// the vector range.
///
/// Read as a first-match reader would, since that is the reader a stale
/// duplicate misleads; storing each key once makes every reader agree.
fn assert_footer_describes_layout(path: &str, bytes: &Bytes, entry: &SuperfileEntry) -> (u64, u64) {
    let file_len = bytes.len() as u64;
    let kvs = raw_footer_kvs(bytes);

    let (fts_at, fts_len) = first_region(&kvs, kv::FTS_OFFSET, kv::FTS_LENGTH)
        .unwrap_or_else(|| panic!("{path}: no FTS region"));
    let (vec_at, vec_len) = first_region(&kvs, kv::VEC_OFFSET, kv::VEC_LENGTH)
        .unwrap_or_else(|| panic!("{path}: no vector region"));

    // The vector range lies inside the file, right after the FTS blob —
    // splice order is body, FTS, vector, ids.
    assert!(
        vec_at
            .checked_add(vec_len)
            .is_some_and(|end| end <= file_len),
        "{path}: footer vector range {vec_at}+{vec_len} runs past the {file_len}-byte file"
    );
    assert_eq!(
        vec_at,
        fts_at + fts_len,
        "{path}: vector blob does not start where the FTS blob ends"
    );

    for key in kv::REGION_KEYS {
        let stored = kvs.iter().filter(|(k, _)| k == key).count();
        assert!(stored <= 1, "{path}: footer stores {key} {stored} times");
    }

    // The footer and the manifest name the same regions.
    let offsets = entry
        .subsection_offsets
        .as_ref()
        .unwrap_or_else(|| panic!("{path}: manifest entry has no subsection offsets"));
    assert_eq!(offsets.total_size, file_len, "{path}: manifest file size");
    assert_eq!(offsets.fts, Some((fts_at, fts_len)), "{path}: FTS region");
    assert_eq!(
        offsets.vec,
        Some((vec_at, vec_len)),
        "{path}: vector region"
    );
    (vec_at, vec_len)
}

/// A footer a past reindex left a stale vector region in is found, and a
/// reindex repairs it.
///
/// The fixture is real 0.9.0 output: that release's reindex stored its
/// input's vector region keys ahead of the output's own. The engine reads
/// the last copy, so the table searches correctly and its FTS index is
/// current — nothing about the index says the file needs work. Only the
/// footer does, so the assessment has to read it, and the repair has to
/// rewrite the footer without disturbing the vectors it locates.
#[test]
fn a_vector_region_a_past_reindex_misplaced_is_found_and_repaired() {
    let Some((_tmp, table, root)) = open_corpus("v7_reindexed_vectors") else {
        return;
    };
    let dir = table_dir(&root);
    let probe = probe_embedding();
    let hits_before = vector_hits(&table, &probe);
    assert!(
        !hits_before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert!(before.superfiles > 0, "the fixture has no superfiles");
    assert_eq!(
        before.needing_rewrite, before.superfiles,
        "every misplaced footer needs a rewrite: {before:?}"
    );
    assert!(!before.is_current(), "{before:?}");
    // The copy this engine reads agrees with the manifest, so these are
    // stale duplicates a rewrite removes, not files left for a person.
    assert!(
        before.inconsistent_footers.is_empty(),
        "a stale duplicate was reported as inconsistent: {before:?}"
    );

    // The plan is what the run does, and a footer needs only a layout
    // rewrite: re-analysis would buy nothing the FTS index lacks.
    let plan = table
        .reindex_plan(&ReindexOptions::default())
        .expect("plan");
    assert_eq!(plan.len(), before.superfiles, "{plan:?}");
    assert!(
        plan.iter().all(|p| p.mode == ReindexMode::Rewrite),
        "a footer repair re-analyzed: {plan:?}"
    );

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("repair the misplaced footers");
    assert_eq!(report.rewritten, before.superfiles, "{report:?}");
    assert!(report.inconsistent_footers.is_empty(), "{report:?}");
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    let reader = table.local_handle().reader().expect("reader");
    for entry in reader.manifest().get_all_superfiles() {
        let path = entry.storage_path();
        let bytes = Bytes::from(fs::read(dir.join(&path)).expect("read output"));
        assert_footer_describes_layout(&path, &bytes, entry);
    }
    assert_eq!(
        vector_hits(&table, &probe),
        hits_before,
        "repairing the footer moved the vectors it locates"
    );

    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess the repaired table");
    assert!(after.is_current(), "{after:?}");
    let again = table
        .reindex(&ReindexOptions::default())
        .expect("second run");
    assert_eq!(again.rewritten, 0, "{again:?}");
}
