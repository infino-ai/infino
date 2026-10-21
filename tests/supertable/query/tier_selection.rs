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
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
};
use tempfile::TempDir;

/// Superfiles committed; each is one commit of [`DOCS_PER_SUPERFILE`] rows.
const SUPERFILES: usize = 12;
/// Rows per superfile.
const DOCS_PER_SUPERFILE: usize = 400;
/// `k` values the equality check runs at.
const KS: [usize; 4] = [1, 10, 100, 1000];
/// Words that every document may carry, by decreasing frequency.
const VOCABULARY: [&str; 8] = [
    "the", "data", "search", "index", "vector", "graph", "tensor", "quokka",
];

fn schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "body",
        arrow_schema::DataType::LargeUtf8,
        false,
    )]))
}

/// A deterministic, skewed corpus: word `w` appears in a document with
/// probability falling with its rank, repeated up to three times so term
/// frequencies and lengths vary, and a few documents carry rare pairs.
fn document(global: usize) -> String {
    let mut x = (global as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut words: Vec<&str> = Vec::new();
    for (rank, w) in VOCABULARY.iter().enumerate() {
        // Word `w` appears in one document in `2^rank`, one to three times.
        let odds = 1u64 << rank;
        if next() % odds == 0 {
            for _ in 0..(1 + next() % 3) {
                words.push(w);
            }
        }
    }
    if words.is_empty() {
        words.push(VOCABULARY[0]);
    }
    let padding = (next() % 6) as usize;
    words.extend(std::iter::repeat_n("filler", padding));
    if global.is_multiple_of(97) {
        words.push("quokka");
        words.push("tensor");
    }
    // Two words that never share a document, so their conjunction is empty.
    words.push(if global.is_multiple_of(2) {
        "zebra"
    } else {
        "yak"
    });
    words.join(" ")
}

fn table(dir: &TempDir, tiers: bool) -> Supertable {
    table_with(dir, tiers, None)
}

/// A table with or without tiers, and with each superfile's head per term
/// cut to `depth` when given: at 64 a common term's heads are read block by
/// block and bounded by their tails rather than read whole.
fn table_with(dir: &TempDir, tiers: bool, depth: Option<usize>) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    // Positions, so a phrase query is among the shapes compared.
    let mut fts = FtsConfig::new("body");
    fts.positions = true;
    // Opens one at a time, so how many superfiles a query opens is a
    // property of the bounds and not of the race between the opens in a
    // window.
    let mut options = SupertableOptions::new(schema(), vec![fts], vec![])
        .expect("options")
        .with_storage(storage)
        .with_bound_ordered_open_window(1);
    if !tiers {
        options = options.with_term_tier_entries_per_term(0);
    }
    if let Some(depth) = depth {
        options = options.with_term_head_depth(depth);
    }
    let st = Supertable::create(options).expect("create");
    let mut w = st.writer().expect("writer");
    for s in 0..SUPERFILES {
        let docs: Vec<String> = (0..DOCS_PER_SUPERFILE)
            .map(|i| document(s * DOCS_PER_SUPERFILE + i))
            .collect();
        let col: ArrayRef = Arc::new(LargeStringArray::from(
            docs.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

fn queries() -> Vec<(String, BoolMode)> {
    let mut out = Vec::new();
    for w in VOCABULARY {
        out.push((w.to_owned(), BoolMode::Or));
    }
    let pairs = [
        ("the", "data"),
        ("search", "index"),
        ("data", "vector"),
        ("graph", "tensor"),
        ("quokka", "tensor"),
        ("the", "quokka"),
    ];
    for (a, b) in pairs {
        out.push((format!("{a} {b}"), BoolMode::Or));
        out.push((format!("{a} {b}"), BoolMode::And));
    }
    for q in [
        "the data search",
        "index vector graph",
        "the search tensor quokka",
        "zebra yak",
        "zebra the",
    ] {
        out.push((q.to_owned(), BoolMode::Or));
        out.push((q.to_owned(), BoolMode::And));
    }
    // Shapes the tiers do not serve take the ceiling path; they must still
    // agree, and must not open more than before.
    for q in ["\"quokka tensor\"", "the -data", "+quokka tensor"] {
        out.push((q.to_owned(), BoolMode::Or));
    }
    out
}

/// `(superfile in commit order, row, score)` per hit, the query's work
/// (opens, index entries read, superfiles the index answered for), and the
/// distinct superfiles among the hits. Superfiles are named by their place
/// in id order so two tables built from the same commits compare.
fn run(
    st: &Supertable,
    query: &str,
    mode: BoolMode,
    k: usize,
) -> (Vec<(usize, u32, f32)>, OpStats, usize) {
    let reader = st.reader().expect("reader");
    let mut order: Vec<(i128, infino::supertable::manifest::SuperfileUri)> = reader
        .manifest()
        .superfiles
        .iter()
        .map(|e| (e.id_min, e.uri))
        .collect();
    order.sort();
    let place = |uri: &infino::supertable::manifest::SuperfileUri| {
        order
            .iter()
            .position(|(_, u)| u == uri)
            .expect("a hit names a listed superfile")
    };
    // The reader is minted inside the scope: a reader carries the collector
    // it was created under, so one minted outside meters nothing.
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("body", query, k, Bm25SearchOptions::new().with_mode(mode))
            .expect("search")
    });
    let rows: Vec<(usize, u32, f32)> = hits
        .iter()
        .map(|h| (place(&h.superfile), h.local_doc_id, h.score))
        .collect();
    let holders = rows
        .iter()
        .map(|r| r.0)
        .collect::<std::collections::HashSet<_>>()
        .len();
    (rows, stats, holders)
}

/// Superfiles a query opened.
fn opened(stats: &OpStats) -> u64 {
    stats.score_pruning_survived
}

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
    let (hits, stats, holders) = run(&st, "quokka tensor", BoolMode::And, 10);
    assert!(!hits.is_empty(), "the planted pair matches");
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

/// A document deleted after the index was built is in the tiers but not
/// in the table: a superfile the index answers for leaves it out, as an
/// opened one would.
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
        assert_eq!(opened(&stats), 0, "k={k}: still answered from the index");
    }
}

/// With every superfile's head cut to 64 documents, a common term's head
/// is read a block at a time and the superfiles it cannot rule out are
/// opened — and the answer is still hit for hit the ceiling path's, at
/// every k, for every shape, with fewer opens over the whole matrix. (A
/// single query can open one more: a head cut this short may not hold the
/// best joint document, so the survivors' open order can differ from the
/// ceilings'.) The heads of rare terms stay whole, so their conjunction is
/// still answered in the index.
#[test]
fn heads_cut_to_a_depth_answer_the_same_and_resolve_the_rare() {
    const DEPTH: usize = 64;
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table_with(&with_dir, true, Some(DEPTH));
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
    let (_, stats, _) = run(&with, "the data", BoolMode::And, 10);
    let held = (2 * SUPERFILES * DEPTH) as u64;
    assert!(
        stats.tier_entries <= held,
        "a common pair reads at most the cut heads: {} entries",
        stats.tier_entries
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
            .with_bound_ordered_open_window(1);
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
            let (b, _, _) = run(&without, query, mode, k);
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
            let total = (SUPERFILES_TIED * DOCS_PER_SUPERFILE_TIED) as u64;
            assert!(
                stats.tier_entries < total,
                "{query:?} {mode:?} k={k}: read {} tier entries of at least {total}",
                stats.tier_entries
            );
            // Every superfile holds the same documents, so after the first
            // open every other can only tie the k-th score, and a tie is
            // settled by `_id` order: the first superfile's documents come
            // first, so the rest are not opened. The tiers' bound for a
            // candidate is its score bit for bit, which is what lets the
            // tie be recognised before the open.
            assert_eq!(
                stats.score_pruning_survived, 1,
                "{query:?} {mode:?} k={k}: only the first superfile is opened"
            );
        }
    }
}
