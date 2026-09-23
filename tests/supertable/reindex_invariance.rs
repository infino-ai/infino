// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A reindex repairs the index regions it names and carries the rest
//! across byte for byte.
//!
//! The unit of repair is the FTS blob; everything else in a superfile is
//! meant to be cargo. That is true today, but true *by accident*: the
//! rewrite path is compaction's, chosen for compaction's reasons, and a
//! change made there for compaction's benefit would quietly widen what a
//! migration touches. These tests are what makes such a change a decision
//! instead of a side effect.
//!
//! The id sidecar is the exception these tests found rather than assumed,
//! and it is why [`REPAIRED_REGIONS`] is a list of two. The builder writes
//! the packed sidecar unconditionally, so a rewrite does one of three
//! things to it depending on how old the input is: adds one to a file that
//! predates the sidecar entirely, re-encodes a raw `i128` array into the
//! packed layout, or leaves an already-packed one alone. The ids
//! themselves are unchanged in every case; only their encoding moves.
//!
//! ## Why the region check enumerates instead of listing
//!
//! The obvious version asserts "the vector region matches, the id region
//! matches" and rots the first time a superfile grows a fourth region:
//! the new one is simply not in the list, and the test passes by not
//! looking. Instead the check reads the regions a superfile *declares* in
//! its Parquet key-value metadata and holds every one it is not declared
//! to repair to byte equality. A region added later is covered the day it
//! exists, and a region that legitimately has to change forces whoever
//! changes it to say so here.
//!
//! ## What is compared by bytes, and what is not
//!
//! Only the spliced index regions are a byte claim. The Parquet body is
//! not: these fixtures were encoded by a published release and a rewrite
//! re-encodes them with the current one, so its bytes legitimately differ
//! down to compression and row-group framing. The body is held to its
//! *values* instead — same rows, same order.
//!
//! Distances are the opposite case and are compared exactly. A splice is
//! byte-faithful or it is broken, so a tolerance there would hide the one
//! defect the check exists to catch.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
    time::Duration,
};

use arrow_array::{Array, Decimal128Array, Float32Array, LargeStringArray};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    Connection, ReindexOptions, Supertable, superfile::format::footer::read_kv_metadata,
    supertable::wal::tombstones_codec::decode_sidecar,
};

use super::corpus_shapes::{TABLE, connect_corpus, corpus_dir, open_corpus, superfile_paths};

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
/// The [`VEC_LAYOUT_KEY`] value for cell-directory subsections — the
/// layout a rewrite has the most work to carry across untouched.
const VEC_LAYOUT_MULTI_CELL: &str = "multi_cell_ivf";
/// Footer key holding a superfile's document count, tombstoned rows
/// included.
const N_DOCS_KEY: &str = "inf.n_docs";

/// The regions a rewrite is allowed to change. Everything else is cargo
/// and is held to byte equality.
///
/// The FTS blob is the point of the operation. The id sidecar is not, and
/// is here because a rewrite re-encodes it whether or not anyone asked:
/// the builder packs the sidecar unconditionally, so a file written before
/// the packed layout existed comes out with one regardless. That is a
/// second format migration riding along with the first — the ids
/// themselves are unchanged, which
/// [`a_rewrite_round_trips_the_stored_columns`] is what actually pins.
///
/// A region is added here only with its reason. The point of the list is
/// that everything outside it is held to byte equality, so growing it is
/// how the boundary moves — deliberately, and in writing.
const REPAIRED_REGIONS: &[&str] = &[FTS_REGION, IDS_REGION];

/// Neighbours retrieved by the probe; large enough that a dropped or
/// reordered row shows up, small enough to stay a cheap assertion.
const PROBE_NEIGHBOURS: usize = 16;
/// Dimension of the corpus generators' planted embeddings.
const EMBEDDING_DIM: usize = 16;
/// Off-axis weight in the probe vector, mirroring the generators'
/// `embedding(0)`.
const PROBE_OFF_AXIS: f32 = 0.05;

/// Rows the deletion test removes. Enough to span more than one superfile
/// so the tombstone carry is exercised per file, few enough to keep the
/// predicate readable.
pub(crate) const DELETED_DOCS: usize = 25;

/// One superfile's spliced regions, as `name -> bytes`.
///
/// Discovered from the key-value metadata rather than named here — see
/// this module's header for why that distinction is the whole point.
///
/// A region declared with a zero length is absent rather than empty, so
/// it is left out: including it would assert equality between two files
/// that both simply lack the region.
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

/// Every superfile's regions under `root`, keyed by region name, with the
/// bytes of each sorted so two tables compare independently of which file
/// holds what.
///
/// A rewrite mints a new superfile id, so input and output cannot be
/// paired by name. Comparing the whole table's regions as a sorted
/// collection sidesteps the pairing entirely and still fails on exactly
/// the things that are wrong: a region whose bytes changed, one that went
/// missing, and one that appeared.
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
///
/// A region runs to megabytes, so the assertion cannot simply show them:
/// the useful facts are how many files carried the region, how long each
/// was, and where the first byte diverges.
fn describe_difference(before: &[Vec<u8>], after: &[Vec<u8>]) -> String {
    let lengths = |regions: &[Vec<u8>]| {
        regions
            .iter()
            .map(|r| r.len().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let first_divergence = before
        .iter()
        .zip(after)
        .enumerate()
        .find_map(|(file, (b, a))| {
            b.iter()
                .zip(a)
                .position(|(x, y)| x != y)
                .map(|at| format!("file {file} first differs at byte {at}"))
                .or_else(|| {
                    (b.len() != a.len())
                        .then(|| format!("file {file} differs in length only, at byte {}", b.len()))
                })
        })
        .unwrap_or_else(|| "a different number of files carry the region".to_owned());
    format!(
        "before: {} region(s) of [{}]\nafter:  {} region(s) of [{}]\n{first_divergence}",
        before.len(),
        lengths(before),
        after.len(),
        lengths(after),
    )
}

/// The probe used against the corpus's planted embeddings; mirrors the
/// generators' `embedding(0)`.
fn probe_embedding() -> Vec<f32> {
    (0..EMBEDDING_DIM)
        .map(|d| if d == 0 { 1.0 } else { PROBE_OFF_AXIS })
        .collect()
}

/// Ids and distances a vector search returns, in rank order.
///
/// Distances ride along because ids alone do not say the vectors survived:
/// a re-encoded index can return the same neighbours with drifted scores,
/// which is precisely the silent damage a splice is supposed to make
/// impossible.
fn vector_hits(table: &Supertable, probe: &[f32]) -> Vec<(i128, f32)> {
    let batches = table
        .vector_search("emb", probe, PROBE_NEIGHBOURS, None, None)
        .expect("vector search");
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        let scores = batch
            .column_by_name("score")
            .expect("score column")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score is f32");
        for i in 0..batch.num_rows() {
            out.push((ids.value(i), scores.value(i)));
        }
    }
    out
}

/// Tombstone the first `DELETED_DOCS` rows by id order and return their
/// `_id`s.
///
/// By id order rather than by any property of the text, so the choice does
/// not depend on which superfile happens to hold what. Shared because
/// several tests — here and in the crash harness — need the same table
/// state, and a second copy of the predicate would be a second chance to
/// delete something different.
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

/// Every row's id and text columns, ordered by id.
///
/// The Parquet body's value-level counterpart to the byte check: a
/// rewrite re-encodes these bytes legitimately, so what has to hold is
/// that the rows come back identical and in the same order.
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

/// **The check that turns "repairs the index and nothing else" from a
/// sentence into an invariant.**
///
/// Every region a superfile declares, except the ones a rewrite is
/// declared to repair, comes out byte-identical. The FTS blob is asserted
/// to have changed, for the same reason the rewritten count is: a check
/// where nothing moved would pass on an engine that did nothing at all.
///
/// A region this engine grows later is carried by default — it is not in
/// [`REPAIRED_REGIONS`], so it must survive untouched, and whoever makes
/// it legitimately change has to come here and say why.
/// Run against every shape rather than one: the carried regions differ by
/// shape — only the hybrid shape has a vector subsection to carry — and a
/// claim about what a rewrite leaves alone is worth exactly as much as the
/// number of writers it has been checked against.
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

    // Only the carried regions have to be the same *set*. A repaired
    // region is allowed to appear: the oldest shapes predate the stable-id
    // sidecar entirely and resolve `_id` from the Parquet id pages, so a
    // rewrite gives them a sidecar they never had. Holding the whole key
    // set equal would call that a defect, and it is the opposite — the
    // file gains an `_id` resolve that decodes no Parquet page.
    let carried_after: Vec<&String> = after
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();
    assert_eq!(
        carried, carried_after,
        "{shape}: a rewrite added or dropped a region it does not repair"
    );
    // Every carried region is reported, not just the first to fail: which
    // ones moved is the finding, and stopping at one hides the rest.
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

// No `v1_positionless` case. A v1-era catalog record names no analyzer,
// so the table cannot be opened at all, let alone reindexed — the refusal
// is pinned by `corpus_shapes::v1_positionless`. v1 is outside the claim
// this module makes rather than an exception to it.

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
        after.len(),
        before.len(),
        "a rewrite changed how many rows the table holds"
    );
    assert_eq!(
        after, before,
        "a rewrite changed the stored columns or the order they come back in"
    );
}

/// The id sidecar is upgraded, not merely disturbed.
///
/// Pinned because the invariance check above permits the ids region to
/// change and a permission granted without a matching assertion is how a
/// region quietly starts changing for the wrong reason. The fixtures
/// predate the packed layout, so they carry the raw `i128` array in — no
/// layout key — and must come out naming the packed layout, smaller.
///
/// That the *ids themselves* survive is not this test's claim; it is
/// [`a_rewrite_round_trips_the_stored_columns`], which compares every
/// row's `_id` in order across the whole table.
#[test]
fn a_rewrite_upgrades_the_id_sidecar_to_the_packed_layout() {
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };

    let layouts = |root: &Path| -> Vec<Option<String>> {
        superfile_paths(root)
            .iter()
            .map(|path| {
                let bytes = fs::read(path).expect("read superfile");
                read_kv_metadata(&bytes)
                    .expect("read superfile key-value metadata")
                    .get(IDS_LAYOUT_KEY)
                    .cloned()
            })
            .collect()
    };

    let before_layouts = layouts(&root);
    assert!(
        before_layouts.iter().all(Option::is_none),
        "the fixture already names an id-sidecar layout, so it cannot show \
         the upgrade: {before_layouts:?}"
    );
    let before_bytes = total_ids_bytes(&root);

    rewrite(&table);

    let after_layouts = layouts(&root);
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

/// The hybrid fixture's vector subsections really are the multi-cell
/// layout, so the byte-identity result above is a claim about the layout
/// that is hardest to carry — not only the single-cell one.
///
/// Asserted rather than assumed: "the vector region survives" is worth
/// what the fixture behind it is worth, and a fixture that quietly
/// regenerated as single-cell would weaken every vector claim in this
/// module without failing any of them.
#[test]
fn the_hybrid_fixture_carries_multi_cell_vector_subsections() {
    let Some(dir) = corpus_dir(SHAPE) else {
        return;
    };

    let layouts: Vec<Option<String>> = superfile_paths(&dir)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            read_kv_metadata(&bytes)
                .expect("read superfile key-value metadata")
                .get(VEC_LAYOUT_KEY)
                .cloned()
        })
        .collect();

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
/// The interaction nothing else covers: deletions live in a per-superfile
/// tombstone sidecar keyed by local doc id, and a rewrite mints a new
/// superfile id. Getting that wrong brings dead rows back, which is worse
/// than the recall loss the migration exists to repair and is invisible to
/// any test that only asks whether the live rows are still there.
///
/// Pins today's behaviour, where the build applies the deletion bitmap and
/// the dead rows are dropped from the output entirely. A change to carry
/// the row set instead has to keep every assertion here passing — the
/// deleted rows stay gone either way, which is the part that matters.
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

/// The rewrite carries the tombstones rather than dropping the rows.
///
/// [`a_rewrite_keeps_deleted_rows_deleted`] proves the *outcome*; this
/// proves the *mechanism*, and the two fail in different places. A build
/// that quietly went back to dropping dead rows would still keep them
/// unreadable and pass that test — while renumbering every surviving row,
/// which is what makes the vector subsection a byte copy instead of a
/// remapping. So: the row count is unchanged, and a sidecar exists under
/// the new superfile id carrying exactly the bits the old one had.
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
    superfile_paths(root)
        .iter()
        .map(|path| {
            let bytes = fs::read(path).expect("read superfile");
            read_kv_metadata(&bytes)
                .expect("read superfile key-value metadata")
                .get(N_DOCS_KEY)
                .expect("every superfile records its document count")
                .parse::<u64>()
                .expect("document count is a number")
        })
        .sum()
}

/// Set bits in each tombstone sidecar under `root`, in path order.
///
/// Read off storage rather than through the cache: the point is what was
/// durably written for the superfile that now exists, not what a reader
/// happens to have resolved.
fn tombstone_bit_counts(root: &Path) -> Vec<u64> {
    let mut counts = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut paths = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read table dir") {
            let path = entry.expect("dir entry").path();
            match path.is_dir() {
                true => stack.push(path),
                false if path.extension().is_some_and(|e| e == "tombstones") => paths.push(path),
                false => {}
            }
        }
    }
    paths.sort();
    for path in paths {
        let bytes = fs::read(&path).expect("read sidecar");
        let sidecar = decode_sidecar(&bytes).expect("decode sidecar");
        if !sidecar.bitmap.is_empty() {
            counts.push(sidecar.bitmap.len());
        }
    }
    counts
}

/// A migration still terminates on a table with deletions.
///
/// Carrying the row set changes what a rewrite produces, and what a
/// rewrite produces is exactly what the planner selects on — so this is
/// the property most at risk from the change and least visible when it
/// breaks. A run that left its output as stale as its input would rewrite
/// the same files on every pass, forever, reporting success each time.
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
