// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! The corpus, tables and runner the term-index tier tests share: a table
//! of several superfiles on storage with a planted vocabulary, built with
//! or without tiers, and a ranked search metered for what it opened and
//! read. Included by path from more than one test binary, so not every
//! item is used by each.
#![allow(dead_code)]

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use infino::{
    Bm25SearchOptions, BoolMode,
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
};
use tempfile::TempDir;

/// Superfiles committed; each is one commit of [`DOCS_PER_SUPERFILE`] rows.
pub const SUPERFILES: usize = 12;
/// Rows per superfile.
pub const DOCS_PER_SUPERFILE: usize = 400;
/// `k` values the equality check runs at.
pub const KS: [usize; 4] = [1, 10, 100, 1000];
/// Words that every document may carry, by decreasing frequency.
pub const VOCABULARY: [&str; 8] = [
    "the", "data", "search", "index", "vector", "graph", "tensor", "quokka",
];

pub fn schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "body",
        arrow_schema::DataType::LargeUtf8,
        true,
    )]))
}

/// A deterministic, skewed corpus: word `w` appears in a document with
/// probability falling with its rank, repeated up to three times so term
/// frequencies and lengths vary, and a few documents carry rare pairs.
pub fn document(global: usize) -> String {
    let mut next =
        infino::test_helpers::xorshift((global as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut words: Vec<&str> = Vec::new();
    for (rank, w) in VOCABULARY.iter().enumerate() {
        // Word `w` appears in one document in `2^rank`, one to three times.
        let odds = 1u64 << rank;
        if next().is_multiple_of(odds) {
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
    // One of sixty-one words, so a query can name sixty-four distinct
    // terms every superfile holds.
    let mut text = words.join(" ");
    text.push_str(&format!(" w{:02}", global % 61));
    text
}

pub fn table(dir: &TempDir, tiers: bool) -> Supertable {
    table_with(
        dir,
        Build {
            tiers,
            ..Build::default()
        },
    )
}

/// How a test table is built.
#[derive(Clone, Copy, Default)]
pub struct Build {
    /// Whether the term index keeps tiers.
    pub tiers: bool,
    /// Each superfile's head per term cut to this depth, when given.
    pub depth: Option<usize>,
    /// A cap on a term's entries across superfiles, when given.
    pub budget: Option<u64>,
    /// Whether the merge is the table's learned call (on) or runs for every
    /// query the tiers can serve (off, what a test of the merge wants).
    pub gate: bool,
    /// Whether every third document's body is null.
    pub nulls: bool,
    /// Whether the table keeps no term index at all (`tiers`, `depth` and
    /// `budget` then have nothing to shape).
    pub without_index: bool,
}

/// The options a test table is created and reopened with.
pub fn options_for(dir: &TempDir, build: Build) -> SupertableOptions {
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
        .with_bound_ordered_open_window(1)
        .with_tier_merge_gate(build.gate);
    if !build.tiers {
        options = options.with_term_tier_entries_per_term(0);
    } else if let Some(budget) = build.budget {
        options = options.with_term_tier_entries_per_term(budget);
    }
    if let Some(depth) = build.depth {
        options = options.with_term_head_depth(depth);
    }
    if build.without_index {
        options = options.with_term_index(false);
    }
    options
}

/// A table with or without tiers, and with each superfile's head per term
/// cut to `depth` when given: at 64 a common term's heads are read block by
/// block and bounded by their tails rather than read whole.
pub fn table_with(dir: &TempDir, build: Build) -> Supertable {
    let st = Supertable::create(options_for(dir, build)).expect("create");
    let mut w = st.writer().expect("writer");
    for s in 0..SUPERFILES {
        let docs: Vec<String> = (0..DOCS_PER_SUPERFILE)
            .map(|i| document(s * DOCS_PER_SUPERFILE + i))
            .collect();
        let col: ArrayRef = Arc::new(LargeStringArray::from(
            docs.iter()
                .enumerate()
                .map(|(i, d)| (!build.nulls || i % 3 != 2).then_some(d.as_str()))
                .collect::<Vec<_>>(),
        ));
        w.append(&RecordBatch::try_new(schema(), vec![col]).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

pub fn queries() -> Vec<(String, BoolMode)> {
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
    // Shapes the tiers serve only through a whole required head, or not at
    // all, must still agree and must not open more than before — among
    // them a phrase with a repeated word, whose bound weights every
    // occurrence as the kernel does.
    for q in [
        "\"quokka tensor\"",
        "the -data",
        "+quokka tensor",
        "\"the data the\"",
        "+alpha \"data the data\"",
        "+the \"quokka tensor\"",
        "\"the data\" +search",
        "data \"data the\"",
        "the the",
        "+data data",
    ] {
        out.push((q.to_owned(), BoolMode::Or));
    }
    // Negations beside a required term, in both modes.
    for q in ["+quokka -tensor", "+zebra -yak", "-the +quokka tensor"] {
        out.push((q.to_owned(), BoolMode::Or));
        out.push((q.to_owned(), BoolMode::And));
    }
    // A negated phrase: the heads cannot verify the adjacency that unmakes
    // a match, so the merge must not settle a superfile from them. The
    // planted pair makes the exclusion bite.
    for q in [
        "quokka -\"quokka tensor\"",
        "+the -\"quokka tensor\"",
        "tensor -\"the data\"",
    ] {
        out.push((q.to_owned(), BoolMode::Or));
        out.push((q.to_owned(), BoolMode::And));
    }
    out
}

/// `(superfile in commit order, row, score)` per hit, the query's work
/// (opens, index entries read, superfiles the index answered for), and the
/// distinct superfiles among the hits. Superfiles are named by their place
/// in id order so two tables built from the same commits compare.
///
/// The query runs twice and the second run is metered: the first brings
/// the terms' tier chunks into memory, after which the merge runs before
/// any open and can answer and rule out superfiles. A first query of a
/// term reads its tiers from storage beside the first opens instead, and
/// answers the same; what it opens is not what the tests assert on.
pub fn run(
    st: &Supertable,
    query: &str,
    mode: BoolMode,
    k: usize,
) -> (Vec<(usize, u32, f32)>, OpStats, usize) {
    st.reader()
        .expect("reader")
        .bm25_hits("body", query, k, Bm25SearchOptions::new().with_mode(mode))
        .expect("search");
    run_once(st, query, mode, k)
}

/// [`run`] without the priming ask: this one ask, metered as it comes.
pub fn run_once(
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
pub fn opened(stats: &OpStats) -> u64 {
    stats.score_pruning_survived
}
