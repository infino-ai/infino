// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A reindex repairs the FTS blob and carries every other spliced region
//! across byte for byte.
//!
//! Regions are read from the footer rather than listed here, so one added
//! later is covered by default. The Parquet body is held to byte equality
//! too, and the stored rows to value equality. Distances are compared
//! exactly — a splice is byte-faithful or broken.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
    time::Duration,
};

use arrow_array::{Array, Decimal128Array, LargeStringArray};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    Connection, ReindexOptions, Supertable,
    superfile::format::{footer::read_kv_metadata, kv as footer_kv},
    supertable::wal::tombstones_codec::decode_sidecar,
};

use super::reindex_fixture::{
    StaleTable, Staleness, TABLE, files_with_extension, footer_values, open_stale, probe_embedding,
    repairing, superfile_paths, vector_hits,
};

// Footer keys come from `format::kv`, not from copies: a test that spells
// them itself keeps passing after the writer renames one.
/// Key suffix holding a region's absolute start offset.
const REGION_OFFSET_SUFFIX: &str = ".offset";
/// Key suffix holding a region's byte length.
const REGION_LENGTH_SUFFIX: &str = ".length";
/// The region a reindex exists to repair.
const FTS_REGION: &str = "fts";
/// The stable-id sidecar, which a repair re-encodes as a side effect.
const IDS_REGION: &str = "ids";
/// The [`footer_kv::VEC_LAYOUT`] value for cell-directory subsections.
const VEC_LAYOUT_MULTI_CELL: &str = "multi_cell_ivf";

/// Regions a repair may change; everything else is held to byte equality.
///
/// The FTS blob is the index being repaired; the id sidecar is here because
/// the builder re-packs it unconditionally. A region joins this list only
/// with its reason.
const REPAIRED_REGIONS: &[&str] = &[FTS_REGION, IDS_REGION];

/// Rows the deletion tests remove.
pub(crate) const DELETED_DOCS: usize = 25;

/// One superfile's spliced regions as `name -> bytes`, read from its footer.
///
/// A zero-length region is absent rather than empty, so it is skipped.
fn spliced_regions(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let kv = read_kv_metadata(bytes).expect("read superfile key-value metadata");
    let mut regions = BTreeMap::new();
    for (key, offset) in &kv {
        let Some(name) = key
            .strip_prefix(footer_kv::PREFIX)
            .and_then(|k| k.strip_suffix(REGION_OFFSET_SUFFIX))
        else {
            continue;
        };
        let length_key = format!("{}{name}{REGION_LENGTH_SUFFIX}", footer_kv::PREFIX);
        let length: usize = kv
            .get(&length_key)
            .unwrap_or_else(|| panic!("{key} has no matching {length_key}"))
            .parse()
            .unwrap_or_else(|e| panic!("{length_key} is not a length: {e}"));
        let offset: usize = offset
            .parse()
            .unwrap_or_else(|e| panic!("{key} is not an offset: {e}"));
        if length == 0 {
            continue;
        }
        regions.insert(name.to_owned(), bytes[offset..offset + length].to_vec());
    }
    regions
}

/// Every superfile's regions under `root`, bytes sorted per name, so two
/// tables compare without pairing superfile ids a rewrite has changed.
fn table_regions(root: &Path) -> BTreeMap<String, Vec<Vec<u8>>> {
    let mut by_name: BTreeMap<String, Vec<Vec<u8>>> = BTreeMap::new();
    for path in superfile_paths(root) {
        let bytes = fs::read(&path).expect("read superfile");
        for (name, region) in spliced_regions(&bytes) {
            by_name.entry(name).or_default().push(region);
        }
    }
    for regions in by_name.values_mut() {
        regions.sort();
    }
    by_name
}

/// How two versions of a region differ, in a line a failure can print.
fn describe_difference(before: &[Vec<u8>], after: &[Vec<u8>]) -> String {
    let lengths = |regions: &[Vec<u8>]| regions.iter().map(Vec::len).collect::<Vec<_>>();
    let first = before.iter().zip(after).position(|(b, a)| b != a);
    format!(
        "before {:?}, after {:?}, first differing file {first:?}",
        lengths(before),
        lengths(after)
    )
}

/// Tombstone the first [`DELETED_DOCS`] rows by id order and return their
/// `_id`s, so the choice does not depend on which superfile holds what.
pub(crate) fn delete_leading_rows(db: &Connection, table: &Supertable) -> HashSet<i128> {
    let victims: Vec<(i128, String)> = rows_by_id(db)
        .into_iter()
        .take(DELETED_DOCS)
        .map(|(id, _body, title, _notes)| (id, title))
        .collect();
    assert_eq!(
        victims.len(),
        DELETED_DOCS,
        "the fixture has fewer rows than this test deletes"
    );

    let predicate = victims
        .iter()
        .map(|(_, title)| col("title").eq(lit(title.clone())))
        .reduce(Expr::or)
        .expect("at least one row to delete");
    let stats = table.delete(predicate).expect("delete rows");
    assert_eq!(
        stats.n_tombstoned(),
        DELETED_DOCS,
        "the delete did not tombstone the rows this test is about"
    );
    victims.into_iter().map(|(id, _)| id).collect()
}

/// Every row's id and text columns, ordered by id — the value-level
/// counterpart to the byte check.
pub(crate) fn rows_by_id(db: &Connection) -> Vec<(i128, String, String, Option<String>)> {
    let batches = db
        .query_sql(&format!(
            "SELECT _id, body, title, notes FROM {TABLE} ORDER BY _id"
        ))
        .expect("select the stored columns");
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        let text = |idx: usize| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("text column is LargeUtf8")
        };
        let (body, title, notes) = (text(1), text(2), text(3));
        for i in 0..batch.num_rows() {
            out.push((
                ids.value(i),
                body.value(i).to_owned(),
                title.value(i).to_owned(),
                (!notes.is_null(i)).then(|| notes.value(i).to_owned()),
            ));
        }
    }
    out
}

/// Repair the table in place with the cheapest repair `staleness` needs,
/// then drop the superseded bytes so what remains on disk is what the
/// table now reads.
///
/// Every superfile has to be rewritten: a run that moved nothing would
/// pass every byte-identity check below without carrying anything.
fn repair(fixture: &StaleTable, staleness: Staleness) {
    let table = &fixture.table;
    let report = table
        .reindex(&repairing(staleness))
        .expect("repair the stale table");
    assert_eq!(
        report.rewritten, fixture.superfiles,
        "{staleness:?}: the repair skipped superfiles, so nothing below is being tested: \
         {report:?}"
    );
    table.gc(Duration::ZERO).expect("collect superseded bytes");
}

/// Every region except [`REPAIRED_REGIONS`] comes out byte-identical.
///
/// Run per repair: re-analysis and a layout rewrite each build their own
/// FTS blob and carry everything else.
fn assert_only_repaired_regions_change(staleness: Staleness) {
    let fixture = open_stale(staleness);
    let root = fixture.root();

    let before = table_regions(root);
    assert!(
        before.contains_key(FTS_REGION),
        "{staleness:?}: the fixture declares no FTS region, so this proves nothing: {:?}",
        before.keys().collect::<Vec<_>>()
    );
    let carried: Vec<&String> = before
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();

    repair(&fixture, staleness);
    let after = table_regions(root);

    let carried_after: Vec<&String> = after
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();
    assert_eq!(
        carried, carried_after,
        "{staleness:?}: a repair added or dropped a region it does not repair"
    );
    // Report every moved region, not just the first: stopping at one hides
    // the rest.
    let moved: Vec<String> = carried
        .into_iter()
        .filter(|name| before.get(*name) != after.get(*name))
        .map(|name| {
            let (b, a) = (&before[name], &after[name]);
            format!("{name}:\n{}", describe_difference(b, a))
        })
        .collect();
    assert!(
        moved.is_empty(),
        "{staleness:?}: a repair changed {} region(s) it does not repair — \
         either the change is a defect, or the region belongs in \
         REPAIRED_REGIONS with the reason written down:\n{}",
        moved.len(),
        moved.join("\n")
    );
}

#[test]
fn a_reanalysis_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change(Staleness::Analysis);
}

#[test]
fn a_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change(Staleness::DuplicatedFooter);
}

/// Vector results survive a repair exactly — the same neighbours at the
/// same distances, compared without tolerance.
#[test]
fn a_repair_preserves_vector_ids_and_distances_exactly() {
    let fixture = open_stale(Staleness::Analysis);
    let table = &fixture.table;

    let probe = probe_embedding();
    let before = vector_hits(table, &probe);
    assert!(
        !before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    repair(&fixture, Staleness::Analysis);

    assert_eq!(
        vector_hits(table, &probe),
        before,
        "a repair moved the vector results it is supposed to splice across untouched"
    );
}

/// The stored columns and their ids round-trip a repair unchanged, in
/// the same order.
#[test]
fn a_repair_round_trips_the_stored_columns() {
    let fixture = open_stale(Staleness::Analysis);

    let before = rows_by_id(&fixture.db);
    assert!(!before.is_empty(), "the fixture has no rows");

    repair(&fixture, Staleness::Analysis);

    assert_eq!(
        rows_by_id(&fixture.db),
        before,
        "a repair changed the stored columns or the order they come back in"
    );
}

/// The fixture really is multi-cell, so the byte-identity claim covers
/// the layout hardest to carry. A fixture that wrote single-cell would
/// weaken every vector claim here without failing one.
#[test]
fn the_fixture_carries_multi_cell_vector_subsections() {
    let fixture = open_stale(Staleness::Analysis);

    let layouts = footer_values(fixture.root(), footer_kv::VEC_LAYOUT);

    assert_eq!(layouts.len(), fixture.superfiles, "{layouts:?}");
    assert!(
        layouts
            .iter()
            .all(|l| l.as_deref() == Some(VEC_LAYOUT_MULTI_CELL)),
        "the fixture is not multi-cell throughout, so the vector \
         byte-identity claim covers less than it appears to: {layouts:?}"
    );
}

/// A repair does not resurrect a deleted row.
///
/// Tombstones are keyed by local doc id and a repair mints a new
/// superfile id, so the carry is where dead rows come back.
#[test]
fn a_repair_keeps_deleted_rows_deleted() {
    let fixture = open_stale(Staleness::Analysis);
    let (db, table) = (&fixture.db, &fixture.table);

    let deleted = delete_leading_rows(db, table);
    let live_before: Vec<i128> = rows_by_id(db).into_iter().map(|(id, ..)| id).collect();
    assert!(
        live_before.iter().all(|id| !deleted.contains(id)),
        "a deleted row was still readable before the repair"
    );
    let probe = probe_embedding();
    let vector_before = vector_hits(table, &probe);
    assert!(
        vector_before.iter().all(|(id, _)| !deleted.contains(id)),
        "vector search returned a deleted row before the repair"
    );

    repair(&fixture, Staleness::Analysis);

    let live_after: Vec<i128> = rows_by_id(db).into_iter().map(|(id, ..)| id).collect();
    assert!(
        live_after.iter().all(|id| !deleted.contains(id)),
        "a repair brought a deleted row back to life"
    );
    assert_eq!(
        live_after, live_before,
        "a repair changed which rows are live, or the order they read in"
    );
    let vector_after = vector_hits(table, &probe);
    assert!(
        vector_after.iter().all(|(id, _)| !deleted.contains(id)),
        "vector search returned a deleted row after the repair"
    );
    assert_eq!(
        vector_after, vector_before,
        "a repair moved the vector results of a table with deletions"
    );
}

/// The repair carries the tombstones rather than dropping the rows:
/// unchanged doc count, same bits under the new superfile id. A build that
/// dropped them would still pass [`a_repair_keeps_deleted_rows_deleted`]
/// while renumbering every row.
#[test]
fn a_repair_carries_the_tombstone_sidecar_to_the_new_superfile() {
    let fixture = open_stale(Staleness::Analysis);
    let root = fixture.root();

    delete_leading_rows(&fixture.db, &fixture.table);

    let docs_before = total_docs(root);
    let sidecars_before = tombstone_bit_counts(root);
    let bits_before: u64 = sidecars_before.iter().sum();
    assert_eq!(
        bits_before, DELETED_DOCS as u64,
        "the delete did not land the bits this test is about: {sidecars_before:?}"
    );

    repair(&fixture, Staleness::Analysis);

    assert_eq!(
        total_docs(root),
        docs_before,
        "a repair dropped rows, so the surviving rows renumbered — the \
         tombstones carried onto the output no longer name the same rows"
    );
    let sidecars_after = tombstone_bit_counts(root);
    assert_eq!(
        sidecars_after.iter().sum::<u64>(),
        bits_before,
        "the repair did not carry the same number of tombstone bits: \
         {sidecars_before:?} -> {sidecars_after:?}"
    );
}

/// Documents every superfile under `root` holds, tombstoned included.
fn total_docs(root: &Path) -> u64 {
    footer_values(root, footer_kv::N_DOCS)
        .iter()
        .map(|v| {
            v.as_ref()
                .expect("every superfile records its document count")
                .parse::<u64>()
                .expect("document count is a number")
        })
        .sum()
}

/// Set bits in each tombstone sidecar under `root`, read off storage
/// rather than through the cache.
fn tombstone_bit_counts(root: &Path) -> Vec<u64> {
    files_with_extension(root, "tombstones")
        .iter()
        .filter_map(|path| {
            let bytes = fs::read(path).expect("read sidecar");
            let sidecar = decode_sidecar(&bytes).expect("decode sidecar");
            (!sidecar.bitmap.is_empty()).then(|| sidecar.bitmap.len())
        })
        .collect()
}

/// A repair still terminates on a table with deletions: a run that left
/// its output as stale as its input would rewrite the same files forever.
#[test]
fn a_second_run_over_a_table_with_deletions_has_nothing_to_do() {
    let fixture = open_stale(Staleness::Analysis);
    let table = &fixture.table;

    delete_leading_rows(&fixture.db, table);

    repair(&fixture, Staleness::Analysis);

    let after = table
        .index_staleness(&ReindexOptions::default())
        .expect("assess the repaired table");
    assert!(
        after.is_current(),
        "a repaired table still reports work, so the run would repeat it: {after:?}"
    );

    let second = table
        .reindex(&ReindexOptions::default())
        .expect("a second run is allowed");
    assert_eq!(
        second.rewritten, 0,
        "a second run rewrote superfiles the first had already brought \
         current: {second:?}"
    );
}

/// The Parquet body is carried across byte for byte, not re-encoded.
///
/// Run per repair, since each has its own carrying build.
fn assert_body_carried(staleness: Staleness) {
    let fixture = open_stale(staleness);

    let before = table_bodies(fixture.root());
    assert_eq!(
        before.len(),
        fixture.superfiles,
        "the fixture has no superfiles"
    );

    repair(&fixture, staleness);

    assert_eq!(
        table_bodies(fixture.root()),
        before,
        "{staleness:?}: a repair re-encoded the Parquet body instead of carrying it"
    );
}

#[test]
fn a_reanalysis_carries_the_parquet_body_byte_for_byte() {
    assert_body_carried(Staleness::Analysis);
}

#[test]
fn a_rewrite_carries_the_parquet_body_byte_for_byte() {
    assert_body_carried(Staleness::DuplicatedFooter);
}

/// Each superfile's Parquet body — everything before the first spliced
/// blob — with the bodies sorted so two tables compare without pairing
/// superfile ids a repair has changed.
fn table_bodies(root: &Path) -> Vec<Vec<u8>> {
    let mut bodies: Vec<Vec<u8>> = superfile_paths(root)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            let kv = read_kv_metadata(&bytes).expect("read superfile key-value metadata");
            let fts_at: usize = kv
                .get(footer_kv::FTS_OFFSET)
                .expect("a superfile under test has an FTS blob")
                .parse()
                .expect("offset is a number");
            bytes[..fts_at].to_vec()
        })
        .collect();
    bodies.sort();
    bodies
}
