// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A reindex repairs the FTS blob and carries every other spliced region
//! across byte for byte.
//!
//! Regions are read from the footer rather than listed here, so one added
//! later is covered by default. The Parquet body is compared by values
//! instead: a rewrite re-encodes it, so its bytes legitimately differ.
//! Distances are compared exactly — a splice is byte-faithful or broken.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
    time::Duration,
};

use arrow_array::{Array, Decimal128Array, LargeStringArray};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    Connection, ReindexOptions, Supertable, superfile::format::footer::read_kv_metadata,
    supertable::wal::tombstones_codec::decode_sidecar,
};

use super::corpus_shapes::{
    TABLE, connect_corpus, corpus_dir, files_with_extension, footer_values, open_corpus,
    probe_embedding, superfile_paths, vector_hits,
};

/// The shape every test here runs on: a table carrying both an FTS index
/// and a vector index, so "nothing but FTS moves" has something to move.
const SHAPE: &str = "v6_hybrid";

/// Prefix on every key a superfile uses to declare a spliced region.
const REGION_KEY_PREFIX: &str = "inf.";
/// Key suffix holding a region's absolute start offset.
const REGION_OFFSET_SUFFIX: &str = ".offset";
/// Key suffix holding a region's byte length.
const REGION_LENGTH_SUFFIX: &str = ".length";
/// The region a reindex exists to repair.
const FTS_REGION: &str = "fts";
/// The stable-id sidecar, which a rewrite re-encodes as a side effect.
const IDS_REGION: &str = "ids";
/// Footer key naming the id sidecar's layout; absent means the raw
/// `i128` array that predates the packed one.
const IDS_LAYOUT_KEY: &str = "inf.ids.layout";
/// The [`IDS_LAYOUT_KEY`] value for the packed sidecar.
const IDS_LAYOUT_PACKED: &str = "packed";
/// Footer key naming the vector blob's layout.
const VEC_LAYOUT_KEY: &str = "inf.vec.layout";
/// The [`VEC_LAYOUT_KEY`] value for cell-directory subsections.
const VEC_LAYOUT_MULTI_CELL: &str = "multi_cell_ivf";
/// Footer key holding the FTS blob's start, which is where the Parquet
/// body ends.
const FTS_OFFSET_KEY: &str = "inf.fts.offset";
/// Footer key holding a superfile's document count, tombstoned rows
/// included.
const N_DOCS_KEY: &str = "inf.n_docs";

/// Regions a rewrite may change; everything else is held to byte equality.
///
/// The id sidecar is here because the builder re-packs it unconditionally.
/// A region joins this list only with its reason.
const REPAIRED_REGIONS: &[&str] = &[FTS_REGION, IDS_REGION];

/// Rows the deletion tests remove; enough to span more than one superfile.
pub(crate) const DELETED_DOCS: usize = 25;

/// One superfile's spliced regions as `name -> bytes`, read from its footer.
///
/// A zero-length region is absent rather than empty, so it is skipped.
fn spliced_regions(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let kv = read_kv_metadata(bytes).expect("read superfile key-value metadata");
    let mut regions = BTreeMap::new();
    for (key, offset) in &kv {
        let Some(name) = key
            .strip_prefix(REGION_KEY_PREFIX)
            .and_then(|k| k.strip_suffix(REGION_OFFSET_SUFFIX))
        else {
            continue;
        };
        let length_key = format!("{REGION_KEY_PREFIX}{name}{REGION_LENGTH_SUFFIX}");
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

/// Rewrite the shape's table in place, then drop the superseded bytes so
/// what remains on disk is what the table now reads.
fn rewrite(table: &Supertable) {
    let report = table
        .reindex(&ReindexOptions::default())
        .expect("a container rewrite is available to a hybrid table");
    assert!(
        report.rewritten > 0,
        "the rewrite migrated nothing, so nothing below is being tested: {report:?}"
    );
    table.gc(Duration::ZERO).expect("collect superseded bytes");
}

/// Every region except [`REPAIRED_REGIONS`] comes out byte-identical, and
/// the FTS blob is asserted to have changed — a check where nothing moved
/// would pass on an engine that did nothing.
///
/// Run per shape: the claim is worth as many writers as it is checked on.
fn assert_only_repaired_regions_change(shape: &str) {
    let Some((_tmp, table, root)) = open_corpus(shape) else {
        return;
    };

    let before = table_regions(&root);
    assert!(
        before.contains_key(FTS_REGION),
        "{shape}: the fixture declares no FTS region, so this proves nothing: {:?}",
        before.keys().collect::<Vec<_>>()
    );
    let carried: Vec<&String> = before
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();

    rewrite(&table);
    let after = table_regions(&root);

    // Only carried regions must be the same set: the oldest shapes gain an
    // id sidecar they never had, which is a repair, not a defect.
    let carried_after: Vec<&String> = after
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();
    assert_eq!(
        carried, carried_after,
        "{shape}: a rewrite added or dropped a region it does not repair"
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
        "{shape}: a rewrite changed {} region(s) it does not repair — either \
         the change is a defect, or the region belongs in REPAIRED_REGIONS \
         with the reason written down:\n{}",
        moved.len(),
        moved.join("\n")
    );
    assert_ne!(
        before.get(FTS_REGION),
        after.get(FTS_REGION),
        "{shape}: the FTS region came through unchanged, so the rewrite \
         repaired nothing"
    );
}

// No `v1_positionless` case: a v1-era record names no analyzer, so the
// table cannot be opened at all. `corpus_shapes::v1_positionless` pins it.

#[test]
fn a_positions_region_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change("v2_positions_region");
}

#[test]
fn a_bitset_block_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change("v4_bitset_blocks");
}

#[test]
fn a_positionless_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change("v5_positionless");
}

#[test]
fn a_positional_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change("v5_positional");
}

#[test]
fn a_coarse_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change("v6_positional");
}

#[test]
fn a_hybrid_rewrite_changes_only_what_it_repairs() {
    assert_only_repaired_regions_change(SHAPE);
}

/// Vector results survive a rewrite exactly — the same neighbours at the
/// same distances, compared without tolerance.
#[test]
fn a_rewrite_preserves_vector_ids_and_distances_exactly() {
    let Some((_tmp, table, _root)) = open_corpus(SHAPE) else {
        return;
    };

    let probe = probe_embedding();
    let before = vector_hits(&table, &probe);
    assert!(
        !before.is_empty(),
        "the fixture's vector index returns nothing"
    );

    rewrite(&table);

    assert_eq!(
        vector_hits(&table, &probe),
        before,
        "a rewrite moved the vector results it is supposed to splice across untouched"
    );
}

/// The stored columns and their ids round-trip a rewrite unchanged, in
/// the same order.
#[test]
fn a_rewrite_round_trips_the_stored_columns() {
    let Some((_tmp, db, _root)) = connect_corpus(SHAPE) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open corpus table");

    let before = rows_by_id(&db);
    assert!(!before.is_empty(), "the fixture has no rows");

    rewrite(&table);

    let after = rows_by_id(&db);
    assert_eq!(
        after, before,
        "a rewrite changed the stored columns or the order they come back in"
    );
}

/// The id sidecar is upgraded, not merely disturbed: raw array in, packed
/// and smaller out. That the ids survive is
/// [`a_rewrite_round_trips_the_stored_columns`]'s claim, not this one.
#[test]
fn a_rewrite_upgrades_the_id_sidecar_to_the_packed_layout() {
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };

    let before_layouts = footer_values(&root, IDS_LAYOUT_KEY);
    assert!(
        before_layouts.iter().all(Option::is_none),
        "the fixture already names an id-sidecar layout, so it cannot show \
         the upgrade: {before_layouts:?}"
    );
    let before_bytes = total_ids_bytes(&root);

    rewrite(&table);

    let after_layouts = footer_values(&root, IDS_LAYOUT_KEY);
    assert!(
        after_layouts
            .iter()
            .all(|l| l.as_deref() == Some(IDS_LAYOUT_PACKED)),
        "a rewrite left an id sidecar in a layout other than the packed \
         one: {after_layouts:?}"
    );
    let after_bytes = total_ids_bytes(&root);
    assert!(
        after_bytes < before_bytes,
        "the packed sidecar is no smaller than the raw array it replaced: \
         {before_bytes} -> {after_bytes} bytes"
    );
}

/// Bytes every superfile under `root` spends on its id sidecar.
fn total_ids_bytes(root: &Path) -> usize {
    table_regions(root)
        .get(IDS_REGION)
        .map(|regions| regions.iter().map(Vec::len).sum())
        .unwrap_or_default()
}

/// The hybrid fixture really is multi-cell, so the byte-identity claim
/// covers the layout hardest to carry. A fixture that regenerated as
/// single-cell would weaken every vector claim here without failing one.
#[test]
fn the_hybrid_fixture_carries_multi_cell_vector_subsections() {
    let Some(dir) = corpus_dir(SHAPE) else {
        return;
    };

    let layouts = footer_values(&dir, VEC_LAYOUT_KEY);

    assert!(!layouts.is_empty(), "the hybrid fixture has no superfiles");
    assert!(
        layouts
            .iter()
            .all(|l| l.as_deref() == Some(VEC_LAYOUT_MULTI_CELL)),
        "the hybrid fixture is not multi-cell throughout, so the vector \
         byte-identity claim covers less than it appears to: {layouts:?}"
    );
}

/// A rewrite does not resurrect a deleted row.
///
/// Tombstones are keyed by local doc id and a rewrite mints a new
/// superfile id, so the carry is where dead rows come back.
#[test]
fn a_rewrite_keeps_deleted_rows_deleted() {
    let Some((_tmp, db, _root)) = connect_corpus(SHAPE) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open corpus table");

    let deleted = delete_leading_rows(&db, &table);
    let live_before: Vec<i128> = rows_by_id(&db).into_iter().map(|(id, ..)| id).collect();
    assert!(
        live_before.iter().all(|id| !deleted.contains(id)),
        "a deleted row was still readable before the rewrite"
    );
    let probe = probe_embedding();
    let vector_before = vector_hits(&table, &probe);
    assert!(
        vector_before.iter().all(|(id, _)| !deleted.contains(id)),
        "vector search returned a deleted row before the rewrite"
    );

    rewrite(&table);

    let live_after: Vec<i128> = rows_by_id(&db).into_iter().map(|(id, ..)| id).collect();
    assert!(
        live_after.iter().all(|id| !deleted.contains(id)),
        "a rewrite brought a deleted row back to life"
    );
    assert_eq!(
        live_after, live_before,
        "a rewrite changed which rows are live, or the order they read in"
    );
    let vector_after = vector_hits(&table, &probe);
    assert!(
        vector_after.iter().all(|(id, _)| !deleted.contains(id)),
        "vector search returned a deleted row after the rewrite"
    );
    assert_eq!(
        vector_after, vector_before,
        "a rewrite moved the vector results of a table with deletions"
    );
}

/// The rewrite carries the tombstones rather than dropping the rows:
/// unchanged doc count, same bits under the new superfile id. A build that
/// went back to dropping them would still pass
/// [`a_rewrite_keeps_deleted_rows_deleted`] while renumbering every row.
#[test]
fn a_rewrite_carries_the_tombstone_sidecar_to_the_new_superfile() {
    let Some((_tmp, db, root)) = connect_corpus(SHAPE) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open corpus table");

    delete_leading_rows(&db, &table);

    let docs_before = total_docs(&root);
    let sidecars_before = tombstone_bit_counts(&root);
    let bits_before: u64 = sidecars_before.iter().sum();
    assert_eq!(
        bits_before, DELETED_DOCS as u64,
        "the delete did not land the bits this test is about: {sidecars_before:?}"
    );

    rewrite(&table);

    assert_eq!(
        total_docs(&root),
        docs_before,
        "a rewrite dropped rows, so the surviving rows renumbered — the \
         tombstones carried onto the output no longer name the same rows"
    );
    let sidecars_after = tombstone_bit_counts(&root);
    assert_eq!(
        sidecars_after.iter().sum::<u64>(),
        bits_before,
        "the rewrite did not carry the same number of tombstone bits: \
         {sidecars_before:?} -> {sidecars_after:?}"
    );
}

/// Documents every superfile under `root` holds, tombstoned included.
fn total_docs(root: &Path) -> u64 {
    footer_values(root, N_DOCS_KEY)
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

/// A migration still terminates on a table with deletions: a run that left
/// its output as stale as its input would rewrite the same files forever.
#[test]
fn a_second_run_over_a_table_with_deletions_has_nothing_to_do() {
    let Some((_tmp, db, _root)) = connect_corpus(SHAPE) else {
        return;
    };
    let table = db.open_table(TABLE).expect("open corpus table");

    delete_leading_rows(&db, &table);

    rewrite(&table);

    let after = table.index_staleness().expect("assess the rewritten table");
    assert_eq!(
        after.needing_rewrite, 0,
        "a rewritten table still reports containers to rewrite, so the \
         migration would repeat this work on every run: {after:?}"
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
/// This is both a correctness claim and the proof that the carrying build
/// ran at all: the merge path re-encodes the body with the current writer,
/// so a fixture written by an older release could not come back identical.
#[test]
fn a_rewrite_carries_the_parquet_body_byte_for_byte() {
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };

    let before = table_bodies(&root);
    assert!(!before.is_empty(), "the fixture has no superfiles");

    rewrite(&table);

    assert_eq!(
        table_bodies(&root),
        before,
        "a rewrite re-encoded the Parquet body instead of carrying it"
    );
}

/// Each superfile's Parquet body — everything before the first spliced
/// blob — with the bodies sorted so two tables compare without pairing
/// superfile ids a rewrite has changed.
fn table_bodies(root: &Path) -> Vec<Vec<u8>> {
    let mut bodies: Vec<Vec<u8>> = superfile_paths(root)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            let kv = read_kv_metadata(&bytes).expect("read superfile key-value metadata");
            let fts_at: usize = kv
                .get(FTS_OFFSET_KEY)
                .expect("a superfile under test has an FTS blob")
                .parse()
                .expect("offset is a number");
            bytes[..fts_at].to_vec()
        })
        .collect();
    bodies.sort();
    bodies
}
