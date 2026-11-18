// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A table shaped like an hourly page-view log: one row per (title, hour),
//! written in time order, every superfile a couple of hours.
//!
//! The shape is what makes it hard to prune. A title viewed every hour puts
//! its terms in every superfile, so routing excludes nothing; and the best
//! document for a term is the same title in every hour, so the top hits tie
//! exactly across superfiles and only the `_id` order separates them. The
//! fixture keeps more superfiles than one open window, as a large table
//! does, and every test compares what the tiers answer against the same
//! table without them.
#![deny(clippy::unwrap_used)]

use std::{ops::Range, sync::Arc};

use arrow::util::display::array_value_to_string;
use arrow_array::{ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    Bm25SearchOptions,
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions, manifest::SuperfileUri},
};
use tempfile::TempDir;

/// Superfiles in the table: well past one open window.
pub const SUPERFILES: usize = 48;
/// Hours each superfile holds.
pub const HOURS_PER_SUPERFILE: usize = 2;
/// Superfiles a ranked query opens at once: a small window, so the table
/// spans many of them as a large table spans its window.
pub const WINDOW: usize = 4;
/// Filler pages per hour that carry the common words, enough for those
/// words' heads to span several blocks in every superfile.
pub const FILLER_PER_HOUR: usize = 90;
/// Hours the rare monument is viewed in, from the first.
pub const MONUMENT_HOURS: usize = 20;
/// Hours, from the first, whose rows of the tied title a test deletes:
/// more than the superfiles holding the top ten span.
const DELETED_HOURS: i64 = 12;

/// Titles viewed in every hour, in both languages: their best documents tie
/// across every superfile.
pub const POPULAR: [&str; 12] = [
    "Zyzzyva",
    "Python",
    "Python (programming language)",
    "Barack Obama",
    "Obama",
    "Michelle Obama",
    "Harry Potter",
    "Quantum chromodynamics",
    "Quantum mechanics",
    "Taylor Swift",
    "Music",
    "Film",
];

/// Languages each popular title is viewed in.
pub const LANGS: [&str; 2] = ["en", "de"];

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("lang", DataType::Utf8, false),
        Field::new("hits", DataType::Int64, false),
        Field::new("hour", DataType::Int64, false),
    ]))
}

/// One hour's rows as `(title, lang, hits)`.
pub fn hour_rows(hour: usize) -> Vec<(String, &'static str, i64)> {
    let mut rows = Vec::new();
    for (t, title) in POPULAR.iter().enumerate() {
        for (l, lang) in LANGS.iter().enumerate() {
            let hits = ((hour * 31 + t * 7 + l * 3) % 2000) as i64;
            rows.push(((*title).to_owned(), *lang, hits));
        }
    }
    if hour < MONUMENT_HOURS {
        rows.push(("Hunnestad Monument".to_owned(), "en", 1));
    }
    for i in 0..FILLER_PER_HOUR {
        let title = match i % 3 {
            0 => format!("List of the music of page {hour} {i}"),
            1 => format!("Of the film archive {hour} {i}"),
            _ => format!("Page {hour} {i} of the united states"),
        };
        rows.push((title, LANGS[i % 2], (i % 50) as i64));
    }
    rows
}

/// How a fixture table is built.
#[derive(Clone, Copy)]
pub struct Shape {
    /// Whether the term index keeps tiers (the merge's input).
    pub tiers: bool,
    /// Each superfile's head per term cut to this depth, when given: a
    /// common term's head then leaves documents unread, as on a large
    /// table where a head holds the best few hundred of millions.
    pub depth: Option<usize>,
    /// The ranked open window.
    pub window: usize,
    /// Whether the table keeps a term index at all.
    pub index: bool,
}

pub fn options(dir: &TempDir, shape: Shape) -> SupertableOptions {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let mut fts = FtsConfig::new("title");
    fts.positions = true;
    let mut options = SupertableOptions::new(schema(), vec![fts], vec![])
        .expect("options")
        .with_storage(storage)
        .with_bound_ordered_open_window(shape.window);
    if !shape.tiers {
        options = options.with_term_tier_entries_per_term(0);
    }
    if let Some(depth) = shape.depth {
        options = options.with_term_head_depth(depth);
    }
    if !shape.index {
        options = options.with_term_index(false);
    }
    options
}

/// The table: one commit per superfile, hours ascending.
pub fn table(dir: &TempDir, shape: Shape) -> Supertable {
    let st = Supertable::create(options(dir, shape)).expect("create");
    commit_superfiles(&st, 0..SUPERFILES);
    st
}

/// Commit superfiles `range` of the table's hours, one commit each.
pub fn commit_superfiles(st: &Supertable, range: Range<usize>) {
    let mut w = st.writer().expect("writer");
    for s in range {
        let mut titles = Vec::new();
        let mut langs = Vec::new();
        let mut hits = Vec::new();
        let mut hours = Vec::new();
        for h in s * HOURS_PER_SUPERFILE..(s + 1) * HOURS_PER_SUPERFILE {
            for (title, lang, n) in hour_rows(h) {
                titles.push(title);
                langs.push(lang);
                hits.push(n);
                hours.push(h as i64);
            }
        }
        let columns: Vec<ArrayRef> = vec![
            Arc::new(LargeStringArray::from(
                titles.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(langs)),
            Arc::new(Int64Array::from(hits)),
            Arc::new(Int64Array::from(hours)),
        ];
        w.append(&RecordBatch::try_new(schema(), columns).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    }
}

/// The ranked answer as `(hour, lang, title, score)` rows, in rank order,
/// with the query's work. Rows, not `_id`s: two tables of the same rows
/// mint different ids, but in the same order, so ties order alike.
pub fn ranked(st: &Supertable, query: &str, k: usize) -> (Vec<String>, OpStats) {
    let (batches, stats) = with_op_stats(|| {
        st.bm25_search(
            "title",
            query,
            k,
            Bm25SearchOptions::new(),
            Some(&["hour", "lang", "title", "score"]),
        )
        .expect("search")
    });
    (rows(&batches), stats)
}

/// Every row of `batches` as one string, columns joined.
pub fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| array_value_to_string(c, i).expect("cell"))
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

/// Superfiles the query opened.
pub fn opened(stats: &OpStats) -> u64 {
    stats.score_pruning_survived
}

/// Two tables of the shape: with tiers, and without as the reference.
pub struct Pair {
    _with_dir: TempDir,
    _without_dir: TempDir,
    pub with: Supertable,
    pub without: Supertable,
}

/// The tiered table shaped by `shape`, and its tierless reference.
pub fn pair(shape: Shape) -> Pair {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, shape);
    let without = table(
        &without_dir,
        Shape {
            tiers: false,
            depth: None,
            ..shape
        },
    );
    Pair {
        _with_dir: with_dir,
        _without_dir: without_dir,
        with,
        without,
    }
}

/// The table past its window, heads whole or cut to `depth`.
pub fn wide(depth: Option<usize>) -> Pair {
    pair(Shape {
        tiers: true,
        depth,
        window: WINDOW,
        index: true,
    })
}

/// Run `query` on both tables: the answers must match exactly; returns the
/// superfiles the tiered table opened.
pub fn opens_and_equal(p: &Pair, query: &str, k: usize) -> u64 {
    let (a, stats) = ranked(&p.with, query, k);
    let (b, _) = ranked(&p.without, query, k);
    assert_eq!(
        a, b,
        "{query:?} k={k}: the tiers must not change the answer"
    );
    opened(&stats)
}

/// Runs of one query a test repeats: enough for a learned gate to have
/// taken each of its ways.
const REPEATS: usize = 4;
/// Opens a tied single term at k = 10 may take: two of its documents sit in
/// every superfile, so five superfiles hold the top ten, and the merge's
/// first opens run a window ahead of its report.
const TIED_TERM_OPENS: u64 = 8;

/// Past one open window the merge runs for every query, whatever the
/// table's learned gate would say: a single term whose best documents tie
/// in every superfile opens a handful on every run, never the table.
#[test]
fn a_table_past_the_open_window_always_merges() {
    let p = wide(None);
    for query in ["python", "zyzzyva", "music"] {
        for run in 0..REPEATS {
            let n = opens_and_equal(&p, query, 10);
            assert!(
                n <= TIED_TERM_OPENS,
                "{query:?} run {run}: opened {n} of {SUPERFILES} superfiles"
            );
        }
    }
}

/// Within one window the learned gate still decides: it takes both its
/// ways, one opening every tied superfile, the other merging and opening a
/// handful. The table holds exactly one window of superfiles, so a merge
/// forced at the window itself would leave only the second.
#[test]
fn a_table_within_the_window_keeps_the_learned_gate() {
    let p = pair(Shape {
        tiers: true,
        depth: None,
        window: SUPERFILES,
        index: true,
    });
    // The first run opens every superfile whatever the gate says: its
    // merge runs beside opens a whole table wide.
    opens_and_equal(&p, "python", 10);
    let opens: Vec<u64> = (1..GATE_REPEATS)
        .map(|_| opens_and_equal(&p, "python", 10))
        .collect();
    assert!(
        opens.iter().any(|&n| n > TIED_TERM_OPENS),
        "{opens:?}: the gate never took its way without the merge"
    );
    assert!(
        opens.iter().any(|&n| n <= TIED_TERM_OPENS),
        "{opens:?}: the gate never merged"
    );
}

/// Runs a test of the learned gate repeats: enough for it to have tried
/// both its ways after the first.
const GATE_REPEATS: usize = 12;

/// Superfiles of the lone-phrase test that hold the whole phrase.
const PHRASE_HOLDERS: usize = 2;

/// A table of [`SUPERFILES`] superfiles past one window: the first
/// [`PHRASE_HOLDERS`] hold "Barack Obama", the rest one of its words only,
/// "Barack Hussein" or "Michelle Obama" in turn. Fewer than ten rows hold
/// the phrase, so a top ten never fills and builds no floor.
fn lone_phrase_table(dir: &TempDir, tiers: bool) -> Supertable {
    let st = Supertable::create(options(
        dir,
        Shape {
            tiers,
            depth: None,
            window: WINDOW,
            index: true,
        },
    ))
    .expect("create");
    let mut w = st.writer().expect("writer");
    for s in 0..SUPERFILES {
        let title = match (s < PHRASE_HOLDERS, s % 2) {
            (true, _) => "Barack Obama",
            (false, 0) => "Barack Hussein",
            (false, _) => "Michelle Obama",
        };
        let titles = vec![title, "Page of the archive"];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(LargeStringArray::from(titles)),
            Arc::new(StringArray::from(vec!["en", "en"])),
            Arc::new(Int64Array::from(vec![1, 1])),
            Arc::new(Int64Array::from(vec![s as i64, s as i64])),
        ];
        w.append(&RecordBatch::try_new(schema(), columns).expect("batch"))
            .expect("append");
        w.commit().expect("commit");
    }
    st
}

/// A query that is one quoted phrase matches only the documents holding
/// every word of it: a superfile holding one word and not the other bounds
/// the phrase at zero through the missing word's empty head, and is ruled
/// out before it is opened. Only the superfiles holding the phrase open.
#[test]
fn a_lone_phrase_rules_out_superfiles_holding_one_of_its_words() {
    let (with_dir, without_dir) = (
        TempDir::new().expect("tempdir"),
        TempDir::new().expect("tempdir"),
    );
    let p = Pair {
        with: lone_phrase_table(&with_dir, true),
        without: lone_phrase_table(&without_dir, false),
        _with_dir: with_dir,
        _without_dir: without_dir,
    };
    // The first ask reads its tiers beside its first opens; the later
    // asks find them resident and merge before any open.
    opens_and_equal(&p, "\"barack obama\"", 10);
    for run in 1..REPEATS {
        let n = opens_and_equal(&p, "\"barack obama\"", 10);
        assert_eq!(
            n, PHRASE_HOLDERS as u64,
            "run {run}: opened {n}, but only {PHRASE_HOLDERS} superfiles hold the phrase"
        );
    }
}

/// Tied rows of a popular title ("Music", "Barack Obama", …) in each
/// superfile: the title in both languages in both its hours.
const MUSIC_PER_SUPERFILE: u64 = (LANGS.len() * HOURS_PER_SUPERFILE) as u64;

/// A term whose best documents tie in every superfile, with heads too
/// shallow to be answered from the index: once the merge has seeded the
/// floor, the superfiles past the ones holding the top `k` are skipped on
/// the tie, by `_id` order. The floor the merge seeds carries the highest
/// `_id` of the superfiles its candidates came from; one that claimed no
/// `_id` would leave every tied superfile to be opened.
#[test]
fn a_merge_seeded_floor_skips_tied_superfiles_by_id() {
    let p = wide(Some(16));
    for query in ["music", "film"] {
        // The first ask reads its tiers beside its first opens; the later
        // asks find them resident and merge before any open.
        opens_and_equal(&p, query, 10);
        for run in 1..REPEATS {
            let n = opens_and_equal(&p, query, 10);
            assert!(
                n <= 10u64.div_ceil(MUSIC_PER_SUPERFILE),
                "{query:?} run {run}: opened {n}, but {} superfiles hold the top 10",
                10u64.div_ceil(MUSIC_PER_SUPERFILE)
            );
        }
    }
}

/// A query that is one quoted phrase, whose holder ties in every
/// superfile: the merge bounds each superfile by the phrase scored as the
/// kernel scores it, so once the first opens have built the floor every
/// superfile past them only ties it and is skipped by `_id` order. The
/// merge seeds no floor of its own (the heads cannot see adjacency), so
/// the opens that build it are those holding the top `k` plus at most one
/// window in flight beside them.
#[test]
fn a_lone_phrase_tied_in_every_superfile_opens_a_window() {
    for depth in [None, Some(16)] {
        let p = wide(depth);
        for query in [
            "\"barack obama\"",
            "\"harry potter\"",
            "\"quantum chromodynamics\"",
        ] {
            opens_and_equal(&p, query, 10);
            for run in 1..REPEATS {
                let n = opens_and_equal(&p, query, 10);
                let most = 10u64.div_ceil(MUSIC_PER_SUPERFILE) + WINDOW as u64;
                assert!(
                    n <= most,
                    "{query} depth {depth:?} run {run}: opened {n} of {SUPERFILES}, at most {most}"
                );
            }
        }
    }
}

/// Query shapes whose candidates the merge cannot trust — a negated term, a
/// required term or phrase, a bare term beside a phrase, a phrase that
/// repeats a word — answer as the table without tiers does, row for row,
/// with heads whole or cut short.
#[test]
fn untrusted_shapes_answer_as_without_tiers() {
    for depth in [None, Some(16)] {
        let p = wide(depth);
        for query in [
            "music -film",
            "python -programming",
            "+python +programming",
            "+obama \"barack obama\"",
            "obama \"barack obama\"",
            "\"obama obama\"",
        ] {
            for _ in 0..REPEATS {
                opens_and_equal(&p, query, 10);
            }
        }
    }
}

/// A `k` past every document the tied term matches: the merge finds no
/// `k`-th candidate, seeds nothing, and every match is returned.
#[test]
fn a_k_past_every_match_answers_as_without_tiers() {
    for depth in [None, Some(16)] {
        let p = wide(depth);
        for (query, k) in [("zyzzyva", 1000), ("monument", 1000), ("music", 500)] {
            for _ in 0..REPEATS {
                opens_and_equal(&p, query, k);
            }
        }
    }
}

/// Rows of the tied title deleted from the first superfiles, which hold
/// the lowest `_id`s the merge's threshold would otherwise draw on: the
/// merge reads past them, and the answer is the table without tiers'.
#[test]
fn deleted_tied_rows_answer_as_without_tiers() {
    for depth in [None, Some(16)] {
        let p = wide(depth);
        for st in [&p.with, &p.without] {
            let mut w = st.writer().expect("writer");
            w.delete(
                col("title")
                    .eq(lit("Music"))
                    .and(col("hour").lt(lit(DELETED_HOURS))),
            )
            .expect("delete");
            w.commit().expect("commit delete");
        }
        for _ in 0..REPEATS {
            opens_and_equal(&p, "music", 10);
        }
    }
}

/// A ranked search under a `WHERE` the index cannot bound, as rows in rank
/// order.
pub fn filtered(st: &Supertable, query: &str, k: usize, filter: &str) -> Vec<String> {
    let batches = st
        .reader()
        .expect("reader")
        .query_sql(&format!(
            "SELECT hour, lang, title, score FROM bm25_search('title', '{query}', {k}) \
             WHERE {filter} ORDER BY score DESC, _id"
        ))
        .expect("filtered search");
    rows(&batches)
}

/// Ranked searches under a `WHERE` the index cannot bound — on the
/// language, on hits, on the hour — answer as the table without tiers
/// does, row for row, whether the predicate keeps most of the top-ranked
/// rows or few of them: each widening round is an ordinary ranked search
/// for more rows, whose merge bounds every superfile soundly, and the
/// search widens until it holds the `k` best rows the predicate keeps.
#[test]
fn filtered_searches_answer_as_without_tiers() {
    for depth in [None, Some(16)] {
        let p = wide(depth);
        for (query, k, filter) in [
            ("python", 10, "lang = 'de'"),
            ("music", 10, "hits > 1990"),
            ("\"barack obama\"", 10, "lang = 'de'"),
            ("zyzzyva", 10, "hour < 3"),
            ("obama", 5, "hour >= 90 AND lang = 'en'"),
        ] {
            for run in 0..2 {
                let a = filtered(&p.with, query, k, filter);
                let b = filtered(&p.without, query, k, filter);
                assert_eq!(a, b, "{query} WHERE {filter} run {run}");
            }
        }
    }
}

/// A prefix search's hits as `(superfile by id order, row, score)`, with
/// the query's work.
pub fn prefix_hits(st: &Supertable, prefix: &str, k: usize) -> (Vec<(usize, u32, f32)>, OpStats) {
    let reader = st.reader().expect("reader");
    let mut order: Vec<(i128, SuperfileUri)> = reader
        .manifest()
        .superfiles
        .iter()
        .map(|e| (e.id_min, e.uri))
        .collect();
    order.sort();
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_search_prefix("title", prefix, k)
            .expect("prefix search")
    });
    let rows = hits
        .iter()
        .map(|h| {
            let place = order
                .iter()
                .position(|(_, u)| *u == h.superfile)
                .expect("a hit names a listed superfile");
            (place, h.local_doc_id, h.score)
        })
        .collect();
    (rows, stats)
}

/// A prefix search expands its prefix from the term index, not from every
/// superfile's dictionary: it opens the superfiles its ranking needs — a
/// prefix of one tied title opens as the title does — where reading the
/// dictionaries opened them all first; and it answers as a table without
/// an index, which reads them.
#[test]
fn a_prefix_expands_from_the_term_index() {
    let p = wide(None);
    let dir = TempDir::new().expect("tempdir");
    let unindexed = table(
        &dir,
        Shape {
            tiers: false,
            depth: None,
            window: WINDOW,
            index: false,
        },
    );
    for (prefix, k) in [("zyzz", 10), ("pyth", 10), ("obam", 5), ("mus", 10)] {
        for run in 0..2 {
            let (a, stats) = prefix_hits(&p.with, prefix, k);
            let (b, _) = prefix_hits(&unindexed, prefix, k);
            assert_eq!(a, b, "{prefix:?} run {run}");
            if prefix == "zyzz" {
                assert!(
                    opened(&stats) <= TIED_TERM_OPENS,
                    "{prefix:?} run {run}: opened {}",
                    opened(&stats)
                );
            }
        }
    }
}

/// Every match of the prefix, so a term the expansion dropped would drop
/// rows; past every document the prefixes below match.
const EVERY_MATCH: usize = 1 << 20;

/// A table whose first superfiles were written with no term index and the
/// rest with one: the index covers only the later superfiles, so a prefix
/// expands from the index for those and from the dictionaries of the
/// earlier ones, and every match of either answers as a table without an
/// index.
#[test]
fn a_prefix_expands_over_superfiles_the_index_does_not_cover() {
    let unindexed_shape = Shape {
        tiers: false,
        depth: None,
        window: WINDOW,
        index: false,
    };
    let dir = TempDir::new().expect("tempdir");
    let early = Supertable::create(options(&dir, unindexed_shape)).expect("create");
    commit_superfiles(&early, 0..SUPERFILES / 2);
    drop(early);
    let mixed = Supertable::open(options(
        &dir,
        Shape {
            index: true,
            ..unindexed_shape
        },
    ))
    .expect("open");
    commit_superfiles(&mixed, SUPERFILES / 2..SUPERFILES);
    let reference_dir = TempDir::new().expect("tempdir");
    let reference = table(&reference_dir, unindexed_shape);
    // The early half has no index: its dictionaries are what a prefix
    // there expands from, so each of those superfiles is opened.
    let (_, stats) = prefix_hits(&mixed, "zyzz", 10);
    assert!(
        opened(&stats) >= (SUPERFILES / 2) as u64,
        "opened {}: the unindexed half reads its dictionaries",
        opened(&stats)
    );
    for prefix in ["zyzz", "pyth", "mus", "of", "monu"] {
        let (a, _) = prefix_hits(&mixed, prefix, EVERY_MATCH);
        let (b, _) = prefix_hits(&reference, prefix, EVERY_MATCH);
        assert!(!a.is_empty(), "{prefix:?} matches");
        assert_eq!(a, b, "{prefix:?}");
    }
}
