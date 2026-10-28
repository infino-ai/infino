// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `hybrid_search` on a table whose vectors live in the hidden vector index.
//!
//! Once a table is drained, its unfiltered vector leg is the search
//! `vector_search` runs: the hidden index plus the undrained user delta. A
//! `WHERE` keeps the leg on the user superfiles. These tests pin
//! what that must preserve, on a table with two drained superfiles and one
//! undrained one:
//!
//!   1. The fused result is exactly reciprocal-rank fusion of the public
//!      `bm25_search` and `vector_search` results at the same `k`, joined on
//!      `_id`: same rows, same scores.
//!   2. Scalar columns resolve to the right row for every fused hit,
//!      including vector-only hits that the hidden index found.
//!   3. A deleted row never comes back, whether it was drained or not, and
//!      the undrained delta is searched.
//!   4. A `WHERE` no index or statistic can narrow returns what the
//!      unfiltered search returns, and a `WHERE` that narrows returns the
//!      best rows that satisfy it.
//!   5. The unfiltered vector leg does exactly the work `vector_search`
//!      does: the same cells, candidates and reranked rows, not a pass over
//!      every user superfile.
//!   6. All of this survives a k past the corpus, a text query no row
//!      matches, an update of a drained row, and a compaction after the
//!      drain.

#![deny(clippy::unwrap_used)]

use std::{collections::HashMap, iter::repeat_n, ops::Range, sync::Arc};

use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    Bm25SearchOptions, BoolMode, OptimizeOptions, VectorSearchOptions,
    runtime_metrics::op_stats::with_op_stats,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_vector_config,
};
use tempfile::TempDir;

/// `default_vector_config` is dim=16, cosine.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 11;
/// One writer thread, so every commit lands as one superfile.
const WRITER_THREADS: usize = 1;
/// Rows per commit: two drained commits, then one undrained.
const DRAINED_COMMITS: [Range<usize>; 2] = [0..6, 6..12];
const UNDRAINED_COMMIT: Range<usize> = 12..16;
/// Top-k for every search here; below the corpus size so fusion truncates.
const TOP_K: usize = 6;
/// Rows in the fixture, across all three commits.
const ROWS: usize = 16;
/// A top-k past every row, so fusion keeps all it is given.
const PAST_THE_CORPUS: usize = 2 * ROWS;
/// A text query no row contains, so the BM25 leg returns nothing.
const ABSENT_TERM: &str = "omega";
/// The text query. Only the rows in [`TERM_TF`] contain it, each a
/// different number of times, so BM25 orders them strictly.
const TERM: &str = "alpha";
/// `(row, times TERM appears)`, spread across all three commits.
const TERM_TF: &[(usize, usize)] = &[(1, 1), (4, 2), (9, 3), (13, 4)];
/// The fused score of one hit at 0-based rank `r` is `1 / (RRF_K + r + 1)`.
const RRF_K: f32 = 60.0;
/// f32 slack when comparing fused scores.
const SCORE_EPS: f32 = 1e-6;

fn fixed_list_f32() -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
    )
}

fn table(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(), false),
    ]));
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(WRITER_THREADS)
            .build()
            .expect("writer pool"),
    );
    let options = SupertableOptions::new(
        schema,
        vec![FtsConfig::new("title")],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
    )
    .expect("valid options")
    .with_writer_pool(pool)
    .with_storage(storage);
    Supertable::create(options).expect("create")
}

/// Row `i`'s title: a unique word, plus [`TERM`] as often as [`TERM_TF`]
/// says.
fn title(row: usize) -> String {
    let tf = TERM_TF
        .iter()
        .find(|(r, _)| *r == row)
        .map_or(0, |(_, tf)| *tf);
    let mut words = vec![format!("row{row}")];
    words.extend(repeat_n(TERM.to_string(), tf));
    words.join(" ")
}

/// Rows `rows`, each with a one-hot embedding at its own dimension.
fn batch(schema: Arc<Schema>, rows: Range<usize>) -> RecordBatch {
    let titles: Vec<String> = rows.clone().map(title).collect();
    let mut flat = Vec::with_capacity(rows.len() * DIM);
    for row in rows {
        flat.extend((0..DIM).map(|d| if d == row { 1.0f32 } else { 0.0 }));
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![Arc::new(LargeStringArray::from(titles)), Arc::new(fsl)],
    )
    .expect("batch")
}

/// Two drained superfiles, then one undrained.
fn drained_table(dir: &TempDir) -> Supertable {
    let st = table(dir);
    let schema = st.options().schema.clone();
    for rows in DRAINED_COMMITS {
        let mut w = st.writer().expect("writer");
        w.append(&batch(schema.clone(), rows)).expect("append");
        w.commit().expect("commit");
    }
    st.drain_vectors_to_cells_sync().expect("drain");
    let mut w = st.writer().expect("writer");
    w.append(&batch(schema, UNDRAINED_COMMIT)).expect("append");
    w.commit().expect("commit");
    st
}

/// A query whose cosine similarity to row `i` grows with `i`, so the vector
/// ranking is strict: row 15 first, then 14, and so on.
fn query_vector() -> Vec<f32> {
    let raw: Vec<f32> = (0..DIM).map(|d| (d + 1) as f32).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.into_iter().map(|x| x / norm).collect()
}

fn csv(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn ids(batches: &[RecordBatch]) -> Vec<i128> {
    let mut out = Vec::new();
    for b in batches {
        let idx = b.schema().index_of("_id").expect("_id");
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal _id");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    out
}

fn f32s(batches: &[RecordBatch], name: &str) -> Vec<f32> {
    let mut out = Vec::new();
    for b in batches {
        let idx = b.schema().index_of(name).expect("column");
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32 column");
        out.extend((0..a.len()).map(|i| a.value(i)));
    }
    out
}

fn strings(batches: &[RecordBatch], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let idx = b.schema().index_of(name).expect("column");
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("utf8 column");
        out.extend((0..a.len()).map(|i| a.value(i).to_string()));
    }
    out
}

/// `_id -> title` for every live row.
fn titles_by_id(st: &Supertable) -> HashMap<i128, String> {
    let rows = st
        .reader()
        .expect("reader")
        .query_sql("SELECT _id, title FROM supertable")
        .expect("scan");
    ids(&rows)
        .into_iter()
        .zip(strings(&rows, "title"))
        .collect()
}

/// Reciprocal-rank fusion of two best-first `_id` lists: every row scoring at
/// least the `k`-th (ties kept, since which of them the engine keeps is its
/// tie-break, not fusion).
fn reference_rrf(bm25: &[i128], vector: &[i128], k: usize) -> HashMap<i128, f32> {
    let mut acc: HashMap<i128, f32> = HashMap::new();
    for list in [bm25, vector] {
        for (rank, id) in list.iter().enumerate() {
            *acc.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut fused: Vec<(i128, f32)> = acc.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    let cutoff = fused.get(k - 1).map_or(f32::MIN, |(_, s)| *s);
    fused
        .into_iter()
        .filter(|(_, s)| *s >= cutoff - SCORE_EPS)
        .collect()
}

/// The public single-leg searches for `term` at `k`, best first.
fn legs(st: &Supertable, term: &str, q: &[f32], k: usize) -> (Vec<i128>, Vec<i128>) {
    let bm25 = st
        .bm25_search(
            "title",
            term,
            k,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
            Some(&["_id"]),
        )
        .expect("bm25_search");
    let vector = st
        .vector_search(
            "emb",
            q,
            k,
            VectorSearchOptions::new(),
            None,
            Some(&["_id"]),
        )
        .expect("vector_search");
    (ids(&bm25), ids(&vector))
}

/// `hybrid_search` through SQL for `term` at `k`, with an optional `WHERE`,
/// projecting `_id`, `title` and `score`.
fn sql_hybrid(st: &Supertable, term: &str, k: usize, filter: &str) -> Vec<RecordBatch> {
    st.reader()
        .expect("reader")
        .query_sql(&format!(
            "SELECT _id, title, score FROM hybrid_search('title', '{term}', 'emb', '{}', {k}) \
             {filter}",
            csv(&query_vector())
        ))
        .expect("hybrid_search tvf")
}

/// Every fused `(id, score)` is the reference's, scores descending, the
/// result holds `k` rows (or every reference row when fewer), and each row's
/// title is its own (so placement found the right row).
fn assert_matches_reference(
    st: &Supertable,
    got: &[RecordBatch],
    reference: &HashMap<i128, f32>,
    k: usize,
) {
    let (got_ids, got_scores) = (ids(got), f32s(got, "score"));
    assert_eq!(got_ids.len(), k.min(reference.len()), "{got_ids:?}");
    for (id, score) in got_ids.iter().zip(&got_scores) {
        let want = reference
            .get(id)
            .unwrap_or_else(|| panic!("fused row {id} is not in the reference {reference:?}"));
        assert!(
            (score - want).abs() < SCORE_EPS,
            "row {id}: fused score {score}, reference {want}"
        );
    }
    for pair in got_scores.windows(2) {
        assert!(pair[0] >= pair[1], "scores must descend: {got_scores:?}");
    }
    if got[0].schema().index_of("title").is_ok() {
        let by_id = titles_by_id(st);
        for (id, title) in got_ids.iter().zip(strings(got, "title")) {
            assert_eq!(
                by_id.get(id),
                Some(&title),
                "row {id} resolved to another row"
            );
        }
    }
}

/// Hybrid through SQL for `term` at `k` is reciprocal-rank fusion of the
/// public legs.
fn assert_sql_hybrid_is_rrf(st: &Supertable, term: &str, k: usize) {
    let (bm25, vector) = legs(st, term, &query_vector(), k);
    let reference = reference_rrf(&bm25, &vector, k);
    assert_matches_reference(st, &sql_hybrid(st, term, k, ""), &reference, k);
}

/// The vector work (cells, candidates, reranked rows) `op` does.
fn vector_work(op: impl FnOnce()) -> (u64, u64, u64) {
    let ((), stats) = with_op_stats(op);
    (
        stats.vector_cells_scanned,
        stats.vector_candidates_scanned,
        stats.vector_rows_reranked,
    )
}

/// The vector work `vector_search` does for the fixture query at [`TOP_K`].
fn vector_search_work(st: &Supertable) -> (u64, u64, u64) {
    let q = query_vector();
    vector_work(|| {
        st.vector_search(
            "emb",
            &q,
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id"]),
        )
        .expect("vector_search");
    })
}

#[test]
fn drained_table_fixture_holds_drained_rows_in_the_hidden_index() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let hidden = st.vector_index_table().expect("hidden index");
    assert_eq!(
        hidden.reader().expect("hidden reader").n_docs_total(),
        DRAINED_COMMITS.iter().map(|r| r.len() as u64).sum::<u64>(),
        "the fixture must hold drained rows in the hidden index"
    );
}

#[test]
fn hybrid_on_a_drained_table_is_rrf_of_the_public_legs() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let q = query_vector();
    let (bm25, vector) = legs(&st, TERM, &q, TOP_K);
    // The fixture must exercise every source: BM25 rows in all three
    // superfiles, vector rows from the undrained delta and the hidden index.
    assert_eq!(bm25.len(), TERM_TF.len());
    assert_eq!(vector.len(), TOP_K);
    let reference = reference_rrf(&bm25, &vector, TOP_K);

    let public = st
        .hybrid_search(
            "title",
            TERM,
            BoolMode::Or,
            "emb",
            &q,
            VectorSearchOptions::new(),
            TOP_K,
            Some(&["_id", "score"]),
        )
        .expect("hybrid_search");
    assert_matches_reference(&st, &public, &reference, TOP_K);
    assert_matches_reference(&st, &sql_hybrid(&st, TERM, TOP_K, ""), &reference, TOP_K);
}

/// A k past every row keeps every row either leg returned, and a text query
/// no row matches leaves the vector leg's ranking alone.
#[test]
fn hybrid_past_the_corpus_and_with_an_empty_text_leg_is_rrf_of_the_legs() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    assert_sql_hybrid_is_rrf(&st, TERM, PAST_THE_CORPUS);
    let (bm25, _) = legs(&st, ABSENT_TERM, &query_vector(), TOP_K);
    assert!(bm25.is_empty(), "the absent term must match no row");
    assert_sql_hybrid_is_rrf(&st, ABSENT_TERM, TOP_K);
}

/// A `WHERE` that admits every row returns what the unfiltered search
/// returns. The filtered leg searches the user superfiles and the unfiltered
/// one the hidden index; on this fixture both return the same rows.
/// `score` is no table column, so nothing can narrow it; `title >= ''` is a
/// statistics test that prunes nothing.
#[test]
fn hybrid_where_that_admits_every_row_matches_the_unfiltered_search() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let unfiltered = sql_hybrid(&st, TERM, TOP_K, "");
    for filter in ["WHERE score > 0", "WHERE title >= ''"] {
        let filtered = sql_hybrid(&st, TERM, TOP_K, filter);
        assert_eq!(ids(&filtered), ids(&unfiltered), "{filter}");
        assert_eq!(
            f32s(&filtered, "score"),
            f32s(&unfiltered, "score"),
            "{filter}"
        );
    }
}

/// A `WHERE` the index narrows to one row returns that row, whether the
/// vector leg found it in a drained or an undrained superfile.
#[test]
fn hybrid_where_that_narrows_returns_only_matching_rows() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let by_id = titles_by_id(&st);
    // Row 9 is drained, row 13 is not.
    for row in [9, 13] {
        let want = title(row);
        let got = sql_hybrid(&st, TERM, TOP_K, &format!("WHERE title = '{want}'"));
        assert_eq!(strings(&got, "title"), vec![want.clone()], "row {row}");
        assert_eq!(
            by_id.get(&ids(&got)[0]),
            Some(&want),
            "row {row} resolved to another row"
        );
    }
}

#[test]
fn hybrid_never_returns_a_deleted_row_drained_or_not() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    // Row 9 is drained, row 13 is not; both rank near the top of both legs.
    let deleted = [title(9), title(13)];
    for t in &deleted {
        let stats = st.delete(col("title").eq(lit(t.as_str()))).expect("delete");
        assert_eq!(stats.n_tombstoned(), 1, "{t}");
    }
    let got = sql_hybrid(&st, TERM, TOP_K, "");
    let got_titles = strings(&got, "title");
    for t in &deleted {
        assert!(!got_titles.contains(t), "deleted row {t} came back");
    }
    assert_sql_hybrid_is_rrf(&st, TERM, TOP_K);
}

/// A drained row updated (deleted, then written again) answers as its new
/// row: the old one, still in the hidden index's cells, never comes back.
#[test]
fn hybrid_returns_an_updated_drained_row_as_its_new_row() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let updated = title(9);
    let old_id = titles_by_id(&st)
        .into_iter()
        .find_map(|(id, t)| (t == updated).then_some(id))
        .expect("row 9 exists");
    st.delete(col("title").eq(lit(updated.as_str())))
        .expect("delete");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&batch(schema, 9..10)).expect("append");
    w.commit().expect("commit");

    let got = sql_hybrid(&st, TERM, TOP_K, "");
    assert!(!ids(&got).contains(&old_id), "the old row came back");
    assert!(
        strings(&got, "title").contains(&updated),
        "the new row is found"
    );
    assert_sql_hybrid_is_rrf(&st, TERM, TOP_K);
}

/// Compacting the user table after the drain moves rows to new superfiles;
/// hybrid still fuses the right rows and resolves each to its own title.
#[test]
fn hybrid_after_a_compaction_past_the_drain_is_rrf_of_the_public_legs() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    st.optimize(&OptimizeOptions::default()).expect("optimize");
    assert_sql_hybrid_is_rrf(&st, TERM, TOP_K);
}

/// The unfiltered vector leg is `vector_search`'s search: the same cells,
/// candidates and reranked rows.
#[test]
fn hybrid_vector_leg_does_the_work_of_vector_search() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_table(&dir);
    let vector = vector_search_work(&st);
    assert!(vector.0 > 0, "the fixture must probe cells");
    let q = query_vector();
    // BM25 adds no vector work, so the counts must agree exactly.
    let hybrid = vector_work(|| {
        st.hybrid_search(
            "title",
            TERM,
            BoolMode::Or,
            "emb",
            &q,
            VectorSearchOptions::new(),
            TOP_K,
            Some(&["_id", "score"]),
        )
        .expect("hybrid_search");
    });
    assert_eq!(
        hybrid, vector,
        "hybrid's vector leg must be vector_search's search"
    );
}
