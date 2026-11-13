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

use std::sync::Arc;

use arrow::util::display::array_value_to_string;
use arrow_array::{ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray};
use infino::{
    Bm25SearchOptions,
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
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

pub fn schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("title", arrow_schema::DataType::LargeUtf8, false),
        arrow_schema::Field::new("lang", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("hits", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("hour", arrow_schema::DataType::Int64, false),
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
    options
}

/// The table: one commit per superfile, hours ascending.
pub fn table(dir: &TempDir, shape: Shape) -> Supertable {
    let st = Supertable::create(options(dir, shape)).expect("create");
    let mut w = st.writer().expect("writer");
    for s in 0..SUPERFILES {
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
    drop(w);
    st
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

/// Within one window the learned gate still decides: it tries the way
/// without the merge, which opens every tied superfile.
#[test]
fn a_table_within_the_window_keeps_the_learned_gate() {
    let p = pair(Shape {
        tiers: true,
        depth: None,
        window: SUPERFILES,
    });
    let most = (0..REPEATS)
        .map(|_| opens_and_equal(&p, "python", 10))
        .max()
        .unwrap_or(0);
    assert!(
        most > TIED_TERM_OPENS,
        "every run opened at most {most}: the gate never took its other way"
    );
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

/// A ranked search under a `WHERE` the index cannot bound, as rows in rank
/// order. (Through the reader's own SQL, which is not metered: what such a
/// search opens is tested where a metered context is reachable.)
pub fn filtered(st: &Supertable, query: &str, k: usize, filter: &str) -> (Vec<String>, OpStats) {
    let (batches, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT hour, lang, title, score FROM bm25_search('title', '{query}', {k}) \
                 WHERE {filter} ORDER BY score DESC, _id"
            ))
            .expect("filtered search")
    });
    (rows(&batches), stats)
}

/// Ranked searches under a `WHERE` the index cannot bound — on the
/// language, on hits, on the hour — answer as the table without tiers
/// does, row for row, whether the predicate keeps most of the top-ranked
/// rows or few of them: the merge under a predicate seeds no threshold of
/// its own, and a search that checks the predicate as each superfile opens
/// still returns the `k` best rows it keeps.
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
                let (a, _) = filtered(&p.with, query, k, filter);
                let (b, _) = filtered(&p.without, query, k, filter);
                assert_eq!(a, b, "{query} WHERE {filter} run {run}");
            }
        }
    }
}
