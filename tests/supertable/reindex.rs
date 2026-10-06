// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reindexing a stale table: what it plans, what it repairs, and that
//! nothing a caller can observe moves.
//!
//! The inputs are this engine's own tables made stale on one axis each
//! (see [`crate::reindex_fixture`]): one records an older analysis
//! revision, which re-analysis repairs; one stores a duplicated footer
//! region key, which a layout rewrite repairs.

use std::{collections::HashMap, fs, time::Duration};

use bytes::Bytes;
use futures::executor::block_on;
use infino::{
    ReindexMode, ReindexOptions, ReindexTarget, SuperfileIndex,
    superfile::{SuperfileReader, VectorSearchOptions, format::kv},
    supertable::manifest::SuperfileEntry,
};

use crate::reindex_fixture::{
    N_DOCS, Staleness, append_docs, assert_scores_equivalent, file_revisions, first_region, hits,
    hits_k, open_stale, probe_embedding, raw_footer_kvs, repairing, scores_by_id, table_dir,
    vector_hits,
};

/// Neighbours a footer-only open is probed for, matching the table-level
/// probe so both see the same depth of the index.
const FOOTER_PROBE_NEIGHBOURS: usize = 16;

/// Documents appended beside the fixture's own to make a mixed table.
const APPENDED_DOCS: u32 = 2;

/// A repair brings every superfile current, changes no answer, and a
/// second run has nothing to do.
fn assert_repair_converges(staleness: Staleness, options: ReindexOptions) {
    let fixture = open_stale(staleness);
    let table = &fixture.table;
    let ranking_before = scores_by_id(table, "body", "common shared", N_DOCS as usize);
    assert!(!ranking_before.is_empty(), "the fixture ranks nothing");
    let probe = probe_embedding();
    let vectors_before = vector_hits(table, &probe);
    assert!(
        !vectors_before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    let report = table.reindex(&options).expect("repair the stale table");
    assert_eq!(
        report.rewritten, fixture.superfiles,
        "{staleness:?}: every stale superfile is rewritten: {report:?}"
    );
    assert_eq!(report.awaiting_reanalysis, 0, "{report:?}");
    assert!(report.unrepairable_columns.is_empty(), "{report:?}");
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    assert!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess the repaired table")
            .is_current(),
        "{staleness:?}: the files still say they are behind"
    );
    assert_scores_equivalent(
        &scores_by_id(table, "body", "common shared", N_DOCS as usize),
        &ranking_before,
        &format!("{staleness:?}"),
    );
    assert_eq!(
        hits(table, "body", "common"),
        N_DOCS as usize,
        "{staleness:?}: the table-wide term stopped matching every document"
    );
    assert_eq!(
        vector_hits(table, &probe),
        vectors_before,
        "{staleness:?}: the repair moved the vector results it carries across"
    );

    // Staleness is read from the files, so a finished run is self-evident
    // and a second one plans nothing.
    let again = table.reindex(&options).expect("a second run");
    assert_eq!(
        again.rewritten, 0,
        "{staleness:?}: reindex is not idempotent"
    );
}

/// Re-analysis records the current revision on every file it rebuilds,
/// so the run terminates.
#[test]
fn reanalysis_repairs_a_table_an_older_analysis_wrote() {
    assert_repair_converges(Staleness::Analysis, ReindexOptions::reanalyzing());
}

/// The default mode resolves an analysis-stale file to re-analysis.
#[test]
fn the_default_mode_repairs_an_analysis_stale_table() {
    assert_repair_converges(Staleness::Analysis, ReindexOptions::default());
}

/// A layout rewrite lays a duplicated footer out afresh.
#[test]
fn a_rewrite_repairs_a_duplicated_footer() {
    assert_repair_converges(Staleness::DuplicatedFooter, ReindexOptions::rewriting());
}

/// A table holding both repaired and unrepaired superfiles reads as if it
/// held neither kind.
///
/// This is the state a reindex passes through on every multi-superfile
/// table, the state an interrupted run leaves, and the one an append
/// creates beside stale files. The corpus statistics a score is
/// normalised by fold over superfiles recording different revisions.
#[test]
fn a_mixed_table_reads_cleanly_across_revisions() {
    let fixture = open_stale(Staleness::Analysis);
    let (table, root) = (&fixture.table, fixture.root());
    let stale_revision = file_revisions(root)[0];

    append_docs(table, N_DOCS..N_DOCS + APPENDED_DOCS);
    let mixed = file_revisions(root);
    let appended = mixed.len() - fixture.superfiles;
    assert!(
        mixed.contains(&stale_revision) && mixed.iter().any(|r| *r > stale_revision),
        "expected both revisions present, got {mixed:?}"
    );

    let with_appended = (N_DOCS + APPENDED_DOCS) as usize;
    assert_eq!(
        hits_k(table, "body", "common", with_appended),
        with_appended,
        "the table-wide term does not span both kinds of superfile"
    );
    let ranking_mixed = scores_by_id(table, "body", "common shared", with_appended);
    assert!(!ranking_mixed.is_empty(), "a mixed table returned nothing");

    // The appended files are current, so the planner skips them.
    let report = table
        .reindex(&ReindexOptions::default())
        .expect("reindex a mixed table");
    assert_eq!(
        report.already_current, appended,
        "the superfiles this engine just wrote were not recognised as current: {report:?}"
    );
    assert_eq!(
        report.rewritten, fixture.superfiles,
        "every stale superfile is rewritten, and only those: {report:?}"
    );
    assert_eq!(
        hits_k(table, "body", "common", with_appended),
        with_appended,
        "the table-wide term stopped spanning the table after the repair"
    );
    assert_scores_equivalent(
        &scores_by_id(table, "body", "common shared", with_appended),
        &ranking_mixed,
        "a mixed table",
    );
}

/// The assessment reports what the run then does — the same numbers, not
/// a parallel estimate of them — and assessing changes nothing.
fn assert_staleness_predicts_the_run(staleness: Staleness) {
    let fixture = open_stale(staleness);
    let table = &fixture.table;
    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess a stale table");

    assert!(!before.is_current(), "{staleness:?}: the fixture is behind");
    assert_eq!(before.superfiles, fixture.superfiles, "{before:?}");
    assert!(before.inconsistent_footers.is_empty(), "{before:?}");
    assert!(before.unrepairable_columns.is_empty(), "{before:?}");
    match staleness {
        Staleness::Analysis => {
            assert_eq!(before.needing_rewrite, 0, "{before:?}");
            assert_eq!(before.awaiting_reanalysis, fixture.superfiles, "{before:?}");
            assert_eq!(
                before.bytes_to_rewrite, 0,
                "no file needs the cheap repair: {before:?}"
            );
        }
        Staleness::DuplicatedFooter => {
            assert_eq!(before.needing_rewrite, fixture.superfiles, "{before:?}");
            assert_eq!(before.awaiting_reanalysis, 0, "{before:?}");
            assert!(
                before.bytes_to_rewrite > 0,
                "a rewrite that moves no bytes is not a rewrite: {before:?}"
            );
        }
    }

    assert_eq!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess again"),
        before,
        "{staleness:?}: assessing the table changed it"
    );

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("repair what the assessment described");
    assert_eq!(
        report.rewritten,
        before.needing_rewrite + before.awaiting_reanalysis,
        "{staleness:?}: the run repaired a different number of files than predicted"
    );
    assert_eq!(
        report.unrepairable_columns, before.unrepairable_columns,
        "{staleness:?}: the run and the assessment name different unrepairable columns"
    );

    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess the repaired table");
    assert!(after.is_current(), "{staleness:?}: {after:?}");
    assert_eq!(after.bytes_to_rewrite, 0, "{after:?}");
    assert_eq!(
        after.superfiles, before.superfiles,
        "{staleness:?}: files appeared or vanished"
    );
}

#[test]
fn staleness_predicts_a_reanalysis_run() {
    assert_staleness_predicts_the_run(Staleness::Analysis);
}

#[test]
fn staleness_predicts_a_rewrite_run() {
    assert_staleness_predicts_the_run(Staleness::DuplicatedFooter);
}

/// A plan names each superfile with the repair its mode resolves to, and
/// running it repairs exactly what it named.
fn assert_plan_resolves(staleness: Staleness, options: ReindexOptions, expected: ReindexMode) {
    let fixture = open_stale(staleness);
    let table = &fixture.table;
    let plan = table.reindex_plan(&options).expect("plan");
    assert_eq!(plan.len(), fixture.superfiles, "{staleness:?}: {plan:?}");
    assert!(
        plan.iter().all(|p| p.mode == expected),
        "{staleness:?} under {options:?} planned other than {expected:?}: {plan:?}"
    );
    assert_eq!(
        table.reindex_plan(&options).expect("plan again"),
        plan,
        "planning changed the table"
    );

    let report = table.reindex(&options).expect("run the plan");
    assert_eq!(report.rewritten, plan.len(), "{report:?}");
    assert!(
        table
            .reindex_plan(&ReindexOptions::default())
            .expect("plan the repaired table")
            .is_empty(),
        "a repaired table has nothing left to plan"
    );
}

/// `Auto` gives each file the cheapest repair that makes it current.
#[test]
fn auto_plans_the_repair_each_staleness_needs() {
    assert_plan_resolves(
        Staleness::Analysis,
        ReindexOptions::default(),
        ReindexMode::Reanalyze,
    );
    assert_plan_resolves(
        Staleness::DuplicatedFooter,
        ReindexOptions::default(),
        ReindexMode::Rewrite,
    );
}

/// `Reanalyze` rebuilds terms even where only the layout is behind, and
/// naming the FTS target explicitly plans what the default target does.
#[test]
fn reanalyze_and_the_fts_target_plan_as_named() {
    assert_plan_resolves(
        Staleness::DuplicatedFooter,
        ReindexOptions::reanalyzing(),
        ReindexMode::Reanalyze,
    );
    assert_plan_resolves(
        Staleness::Analysis,
        ReindexOptions::default().with_target(ReindexTarget::Superfile(SuperfileIndex::Fts)),
        ReindexMode::Reanalyze,
    );
}

/// The default mode leaves nothing stale, where a layout rewrite cannot.
///
/// A rewrite copies postings, so it cannot clear an analysis revision; it
/// plans nothing rather than a rewrite that would repeat on every run, and
/// says what it left.
#[test]
fn the_default_mode_leaves_the_table_current_where_a_rewrite_cannot() {
    let fixture = open_stale(Staleness::Analysis);
    let table = &fixture.table;

    let rewrite = table
        .reindex(&ReindexOptions::rewriting())
        .expect("a rewrite over an analysis-stale table");
    assert_eq!(
        (rewrite.rewritten, rewrite.awaiting_reanalysis),
        (0, fixture.superfiles),
        "a rewrite has to leave the analysis axis behind and say so: {rewrite:?}"
    );
    assert!(
        !table
            .index_staleness(&ReindexOptions::default())
            .expect("assess")
            .is_current(),
        "a rewrite left the table looking finished"
    );

    let report = table
        .reindex(&ReindexOptions::default())
        .expect("the default repairs the table");
    assert_eq!(report.rewritten, fixture.superfiles, "{report:?}");
    assert_eq!(report.awaiting_reanalysis, 0, "{report:?}");
    assert!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess")
            .is_current(),
        "the default mode finished with the table still stale"
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
/// Re-analysis rebuilds the FTS blob, which can change size and move the
/// vector bytes after it. Every region key has to follow them, stored
/// exactly once.
#[test]
fn a_carried_vector_subsection_gets_a_footer_that_describes_it() {
    let fixture = open_stale(Staleness::Analysis);
    let (table, root) = (&fixture.table, fixture.root());
    let dir = table_dir(root);
    let probe = probe_embedding();

    // Inputs keyed by their vector bytes: a carried subsection is copied
    // byte for byte, so that is what pairs an output with its input.
    let mut inputs: HashMap<Bytes, Bytes> = HashMap::new();
    for entry in fs::read_dir(dir.join("data")).expect("read data dir") {
        let bytes = Bytes::from(fs::read(entry.expect("dir entry").path()).expect("read input"));
        let kvs = raw_footer_kvs(&bytes);
        let (at, len) = first_region(&kvs, kv::VEC_OFFSET, kv::VEC_LENGTH)
            .expect("every fixture superfile carries a vector subsection");
        inputs.insert(bytes.slice(at as usize..(at + len) as usize), bytes);
    }
    assert_eq!(
        inputs.len(),
        fixture.superfiles,
        "one vector subsection per input"
    );

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
        let (vec_at, vec_len) = assert_footer_describes_layout(&path, &bytes, entry);

        let vec_bytes = bytes.slice(vec_at as usize..(vec_at + vec_len) as usize);
        let input = inputs
            .get(&vec_bytes)
            .unwrap_or_else(|| panic!("{path}: vector bytes match no input's subsection"));

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

    // Splice order is body, FTS, vector, ids.
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

/// A footer storing a stale vector region key ahead of the real one is
/// found, and a reindex repairs it.
///
/// The engine reads the last copy, so the table searches correctly and
/// its FTS index is current — only the footer says the file needs work.
/// The assessment has to read it, and the repair has to rewrite the
/// footer without disturbing the vectors it locates.
#[test]
fn a_duplicated_vector_region_key_is_found_and_repaired() {
    let fixture = open_stale(Staleness::DuplicatedFooter);
    let (table, root) = (&fixture.table, fixture.root());
    let dir = table_dir(root);
    let probe = probe_embedding();
    let hits_before = vector_hits(table, &probe);
    assert!(
        !hits_before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    let before = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess");
    assert_eq!(
        before.needing_rewrite, fixture.superfiles,
        "every duplicated footer needs a rewrite: {before:?}"
    );
    // The copy this engine reads agrees with the manifest, so these are
    // stale duplicates a rewrite removes, not files left for a person.
    assert!(
        before.inconsistent_footers.is_empty(),
        "a stale duplicate was reported as inconsistent: {before:?}"
    );

    let report = table
        .reindex(&repairing(Staleness::DuplicatedFooter))
        .expect("repair the duplicated footers");
    assert_eq!(report.rewritten, fixture.superfiles, "{report:?}");
    assert!(report.inconsistent_footers.is_empty(), "{report:?}");
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    let reader = table.local_handle().reader().expect("reader");
    for entry in reader.manifest().get_all_superfiles() {
        let path = entry.storage_path();
        let bytes = Bytes::from(fs::read(dir.join(&path)).expect("read output"));
        assert_footer_describes_layout(&path, &bytes, entry);
    }
    assert_eq!(
        vector_hits(table, &probe),
        hits_before,
        "repairing the footer moved the vectors it locates"
    );
}
