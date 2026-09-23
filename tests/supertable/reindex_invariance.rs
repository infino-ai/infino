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
//! and it is why [`REPAIRED_REGIONS`] is a list of two: a rewrite also
//! re-encodes a pre-packed-layout sidecar into the packed one. The ids
//! are unchanged; their encoding is not.
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

use std::{collections::BTreeMap, fs, path::Path, time::Duration};

use arrow_array::{Array, Decimal128Array, Float32Array, LargeStringArray};
use infino::{Connection, ReindexOptions, Supertable, superfile::format::footer::read_kv_metadata};

use super::corpus_shapes::{TABLE, connect_corpus, open_corpus, superfile_paths};

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

/// The regions a rewrite is allowed to change. Everything else is cargo
/// and is held to byte equality.
///
/// The FTS blob is the point of the operation. The id sidecar is not, and
/// is here because a rewrite re-encodes it whether or not anyone asked:
/// the builder writes the packed frame-of-reference layout, so a file
/// written before that layout existed carries a raw `i128` array in and a
/// packed sidecar out. That is a second format migration riding along
/// with the first — the ids themselves are unchanged, which
/// [`a_rewrite_round_trips_the_stored_columns`] is what actually pins.
const REPAIRED_REGIONS: &[&str] = &[FTS_REGION, IDS_REGION];

/// Neighbours retrieved by the probe; large enough that a dropped or
/// reordered row shows up, small enough to stay a cheap assertion.
const PROBE_NEIGHBOURS: usize = 16;
/// Dimension of the corpus generators' planted embeddings.
const EMBEDDING_DIM: usize = 16;
/// Off-axis weight in the probe vector, mirroring the generators'
/// `embedding(0)`.
const PROBE_OFF_AXIS: f32 = 0.05;

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

/// Every row's id and text columns, ordered by id.
///
/// The Parquet body's value-level counterpart to the byte check: a
/// rewrite re-encodes these bytes legitimately, so what has to hold is
/// that the rows come back identical and in the same order.
fn rows_by_id(db: &Connection) -> Vec<(i128, String, String, Option<String>)> {
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
#[test]
fn a_rewrite_changes_only_the_regions_it_repairs() {
    let Some((_tmp, table, root)) = open_corpus(SHAPE) else {
        return;
    };

    let before = table_regions(&root);
    assert!(
        before.contains_key(FTS_REGION),
        "the fixture declares no FTS region, so this proves nothing: {:?}",
        before.keys().collect::<Vec<_>>()
    );
    let carried: Vec<&String> = before
        .keys()
        .filter(|n| !REPAIRED_REGIONS.contains(&n.as_str()))
        .collect();
    assert!(
        !carried.is_empty(),
        "the fixture declares only repaired regions, so there is nothing to carry: {:?}",
        before.keys().collect::<Vec<_>>()
    );

    rewrite(&table);
    let after = table_regions(&root);

    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "a rewrite added or dropped a whole region"
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
        "a rewrite changed {} region(s) it does not repair — either the \
         change is a defect, or the region belongs in REPAIRED_REGIONS \
         with the reason written down:\n{}",
        moved.len(),
        moved.join("\n")
    );
    assert_ne!(
        before.get(FTS_REGION),
        after.get(FTS_REGION),
        "the FTS region came through unchanged, so the rewrite repaired nothing"
    );
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
