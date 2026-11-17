// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The level heads a build and a fold write, against a brute force over
//! every document of the term: what a head leaves out never keys above its
//! floor, every document keying above the floor is in it, its depth
//! follows the layer's document count, and its per-bucket maxima are the
//! true ones.

use std::{cmp::Ordering, collections::BTreeSet, fs, path::Path};

use proptest::prelude::*;
use tempfile::TempDir;
use uuid::Uuid;

use super::{
    build::{
        BuildPolicy, BuiltBytes, Contribution, ContributionWriter, SliceFile, SliceSource, build,
        merge_layers, tier_path,
    },
    format::{Layer, Location, Slice},
    level_head::{BucketMaxTf, LevelBlock, decode_segment, head_depth},
    tier::{ColumnScoring, TierEntry},
};
use crate::{supertable::manifest::part::ContentHash, test_helpers::fid, utils::varint::push_head};

/// Proptest cases per property: each builds and folds whole layers.
const CASES: u32 = 24;
/// Most superfiles a generated layer spans.
const MAX_SUPERFILES: usize = 4;
/// Most documents a generated superfile holds the term in: past the
/// whole-list size, so some layers cut their heads.
const MAX_DOCS: usize = 3000;
/// The term every fixture indexes.
const TERM: &str = "t";
const COLUMN: &str = "body";

/// One document of the term: its superfile ordinal, row, term frequency
/// and length bucket.
type Doc = (u32, u32, u32, u8);

fn key(doc: &Doc) -> f32 {
    ColumnScoring::default().key(doc.2, doc.3)
}

/// Head order: key descending, then ordinal and row ascending.
fn by_head_order(a: &Doc, b: &Doc) -> Ordering {
    key(b).total_cmp(&key(a)).then((a.0, a.1).cmp(&(b.0, b.1)))
}

/// One superfile's contribution for the term: its best `depth` documents
/// as its head, its tail the key of the best one left out.
fn contribution(dir: &Path, ordinal: u32, docs: &[Doc], depth: usize) -> Contribution {
    let mut w = ContributionWriter::create(dir, Uuid::from_u128(u128::from(ordinal) + 1), 0)
        .expect("create");
    w.set_scoring(fid(COLUMN), ColumnScoring::default());
    let mut sorted = docs.to_vec();
    sorted.sort_by(by_head_order);
    let kept = &sorted[..depth.min(sorted.len())];
    let tail = sorted.get(depth).map_or(0.0, key);
    let bound = sorted.first().map_or(0.0, key);
    let mut head = Vec::new();
    push_head(&mut head, kept.iter().map(|d| (d.1, d.2, d.3)));
    let mut buckets = BucketMaxTf::default();
    for d in docs {
        buckets.observe(d.3, d.2);
    }
    w.push_encoded(
        &fid(COLUMN).term_key(TERM),
        docs.len() as u64,
        bound,
        Location::None,
        tail,
        &head,
        Some(buckets.pairs()),
    )
    .expect("push");
    w.finish().expect("finish")
}

/// The term's head in a built layer: its floor, its per-bucket maxima and
/// its entries, level by level in head order.
fn head_of(
    layer: &Layer,
    slices: &[(ContentHash, BuiltBytes)],
    objects: &[(ContentHash, BuiltBytes)],
) -> (
    f32,
    Option<BucketMaxTf>,
    Vec<LevelBlock>,
    Vec<TierEntry>,
    u64,
) {
    assert_eq!(layer.slices.len(), 1, "fixtures build one slice");
    let slice_ref = &layer.slices[0];
    let bytes = slices
        .iter()
        .find(|(h, _)| *h == slice_ref.content_hash)
        .expect("the slice is built")
        .1
        .read()
        .expect("slice bytes");
    let run = Slice::open(&bytes)
        .expect("open")
        .run(&fid(COLUMN).term_key(TERM))
        .expect("run")
        .expect("the term is in the slice");
    let head = run.head.expect("the term has a head");
    let n_entries = head.n_entries();
    let mut blocks = Vec::new();
    if !head.levels.is_empty() {
        let object_hash = slice_ref.tier.expect("the slice has an object");
        let object = objects
            .iter()
            .find(|(h, _)| *h == object_hash)
            .expect("the object is built")
            .1
            .read()
            .expect("object bytes");
        let mut level_blocks = Vec::new();
        for level in &head.levels {
            let range = level.offset as usize..(level.offset + level.len) as usize;
            decode_segment(&object[range], &mut level_blocks).expect("segment");
            assert_eq!(
                level_blocks.iter().map(|b| b.entries.len()).sum::<usize>(),
                level.n_entries as usize
            );
            blocks.append(&mut level_blocks);
        }
    }
    (head.floor, head.buckets, blocks, head.inline, n_entries)
}

/// Check a head against every document of the term (`all`), and the
/// candidates the head was cut from (`candidates`).
fn check_head(
    (floor, buckets, blocks, inline, n_entries): (
        f32,
        Option<BucketMaxTf>,
        Vec<LevelBlock>,
        Vec<TierEntry>,
        u64,
    ),
    all: &[Doc],
    candidates: usize,
) {
    // Blocks descend: each block's keys lie between its header's, and no
    // later block keys above an earlier one's smallest.
    for block in &blocks {
        for e in &block.entries {
            let k = ColumnScoring::default().key(e.tf, e.len);
            assert!(k <= block.first_key && k >= block.last_key);
        }
    }
    for w in blocks.windows(2) {
        assert!(w[0].last_key >= w[1].first_key, "blocks in head order");
    }
    let held: Vec<Doc> = blocks
        .iter()
        .flat_map(|b| b.entries.iter())
        .chain(inline.iter())
        .map(|e| (e.superfile, e.doc_id, e.tf, e.len))
        .collect();
    let held_set: BTreeSet<Doc> = held.iter().copied().collect();
    assert_eq!(held_set.len(), held.len(), "no document twice");
    assert_eq!(held.len() as u64, n_entries);
    let all_set: BTreeSet<Doc> = all.iter().copied().collect();
    assert!(
        held_set.is_subset(&all_set),
        "the head holds real documents"
    );
    // Sound: nothing left out keys above the floor.
    for d in all_set.difference(&held_set) {
        assert!(
            key(d) <= floor,
            "{d:?} left out keys above the floor {floor}"
        );
    }
    // Exact above the floor: every document keying above it is held.
    for d in all.iter().filter(|d| key(d) > floor) {
        assert!(
            held_set.contains(d),
            "{d:?} keys above the floor but is not held"
        );
    }
    // As deep as the layer's document count asks, or every candidate.
    let want = (head_depth(all.len() as u64) as usize).min(candidates);
    assert_eq!(held.len(), want);
    let mut truth = BucketMaxTf::default();
    for d in all {
        truth.observe(d.3, d.2);
    }
    assert_eq!(buckets, Some(truth));
}

/// Generated superfiles: each a list of documents with distinct rows and
/// the depth its own head keeps.
fn arb_superfiles() -> impl Strategy<Value = Vec<(Vec<Doc>, usize)>> {
    prop::collection::vec(
        (
            prop::collection::btree_set(0u32..20_000, 1..MAX_DOCS),
            prop::collection::vec((1u32..12, 0u8..40), MAX_DOCS),
            prop_oneof![1usize..64, Just(usize::MAX), 64usize..MAX_DOCS],
        ),
        1..=MAX_SUPERFILES,
    )
    .prop_map(|superfiles| {
        superfiles
            .into_iter()
            .enumerate()
            .map(|(ordinal, (rows, facts, depth))| {
                let docs = rows
                    .into_iter()
                    .zip(facts)
                    .map(|(row, (tf, len))| (ordinal as u32, row, tf, len))
                    .collect();
                (docs, depth)
            })
            .collect()
    })
}

fn policy() -> BuildPolicy {
    BuildPolicy {
        slice_target_bytes: usize::MAX,
        ..BuildPolicy::default()
    }
}

/// Write a built layer's slices and objects where a merge reads them.
fn place(
    dir: &Path,
    layer: &Layer,
    slices: &[(ContentHash, BuiltBytes)],
    objects: &[(ContentHash, BuiltBytes)],
) -> Vec<SliceFile> {
    let mut files = Vec::new();
    for slice in &layer.slices {
        let path = dir.join(format!("{}.slice", slice.content_hash.to_hex()));
        let bytes = slices
            .iter()
            .find(|(h, _)| *h == slice.content_hash)
            .expect("built slice")
            .1
            .read()
            .expect("slice bytes");
        fs::write(&path, bytes).expect("place slice");
        if let Some(object) = slice.tier {
            let bytes = objects
                .iter()
                .find(|(h, _)| *h == object)
                .expect("built object")
                .1
                .read()
                .expect("object bytes");
            fs::write(tier_path(dir, &slice.content_hash), bytes).expect("place object");
        }
        files.push(SliceFile {
            source: SliceSource::Local(path),
            first_key: slice.first_key.clone(),
            last_key: slice.last_key.clone(),
            base: layer.base,
        });
    }
    files
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// A build cuts the layer's head from its superfiles' heads.
    #[test]
    fn a_built_head_is_sound_and_exact_above_its_floor(superfiles in arb_superfiles()) {
        let dir = TempDir::new().expect("tempdir");
        let contributions: Vec<Contribution> = superfiles
            .iter()
            .enumerate()
            .map(|(o, (docs, depth))| contribution(dir.path(), o as u32, docs, *depth))
            .collect();
        let built = build(&contributions, &policy()).expect("build");
        let all: Vec<Doc> = superfiles.iter().flat_map(|(d, _)| d.iter().copied()).collect();
        let candidates = superfiles.iter().map(|(d, depth)| d.len().min(*depth)).sum();
        check_head(head_of(&built.root.layers[0], &built.slices, &built.tiers), &all, candidates);
    }

    /// A fold of two layers cuts the merged head from the layers' heads,
    /// and stays sound and exact above its floor.
    #[test]
    fn a_folded_head_is_sound_and_exact_above_its_floor(
        superfiles in arb_superfiles(),
        split in 0usize..=MAX_SUPERFILES,
    ) {
        let dir = TempDir::new().expect("tempdir");
        let split = split.min(superfiles.len());
        let mut sources = Vec::new();
        let mut candidates = 0usize;
        for (base, group) in [(0, &superfiles[..split]), (split, &superfiles[split..])] {
            if group.is_empty() {
                continue;
            }
            let contributions: Vec<Contribution> = group
                .iter()
                .enumerate()
                .map(|(o, (docs, depth))| contribution(dir.path(), o as u32, &docs.iter().map(|d| (o as u32, d.1, d.2, d.3)).collect::<Vec<_>>(), *depth))
                .collect();
            let built = build(&contributions, &policy()).expect("build");
            let mut layer = built.root.layers[0].clone();
            layer.place_at(base as u32, group.len() as u32);
            let all: Vec<Doc> = group.iter().flat_map(|(d, _)| d.iter().copied()).collect();
            candidates += (head_depth(all.len() as u64) as usize)
                .min(group.iter().map(|(d, depth)| d.len().min(*depth)).sum());
            sources.push(place(dir.path(), &layer, &built.slices, &built.tiers));
        }
        let ordinals: Vec<Option<u32>> = (0..superfiles.len() as u32).map(Some).collect();
        let merged = merge_layers(&sources, &ordinals, &policy()).expect("merge");
        let all: Vec<Doc> = superfiles.iter().flat_map(|(d, _)| d.iter().copied()).collect();
        check_head(head_of(&merged.layer, &merged.slices, &merged.tiers), &all, candidates);
    }
}
