// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Multi-field BM25: `bm25_search_fields` against the same query answered
//! by single-column `bm25_search` calls merged in the client.
//!
//! A best-fields query scores each clause by its best column and sums the
//! clauses. A client holding only `bm25_search` can answer it two ways:
//!
//! - **client, every match**: search each (clause, column) pair for all of
//!   its matches and merge them. Exact, but it reads and ships every match.
//! - **client, top k per column**: search each pair for its top k and merge
//!   those. It reads less, but a row strong in no single column and
//!   strong overall can be missed, so its top k can differ; the report
//!   gives its overlap with the exact top k.
//!
//! The table has three full-text columns of different lengths (title,
//! body, tags) over one Zipfian vocabulary, written in several commits so
//! the searches span superfiles. Each query is timed warm: the median and
//! p95 of repeated runs on one connection. Every `bm25_search_fields`
//! answer is checked against the exact client merge.
//!
//! Invoked as `cargo bench -- fts-fields`. Knobs:
//! `INFINO_BENCH_FTS_FIELDS_ROWS` (default 500 000),
//! `INFINO_DIAG_ITERS` (warm runs per query, default 20).

use std::{
    collections::HashMap,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Array, Decimal128Array, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::{
    Bm25FieldsOptions, Bm25SearchOptions, BoolMode, FieldClause, IndexSpec, Supertable, connect,
};
use rand::{SeedableRng, rngs::StdRng};
use tempfile::TempDir;

use crate::{
    corpus::ZipfDistribution,
    diag_common::{env_usize, percentile},
    markdown::fmt_time,
    report::{Better, Block, Cell, Report, Section, metric, text},
};

const DEFAULT_ROWS: usize = 500_000;
/// Rows per commit: one superfile each.
const ROWS_PER_COMMIT: usize = 50_000;
const DEFAULT_ITERS: usize = 20;
/// Distinct terms across the three columns.
const VOCAB: usize = 20_000;
/// Terms per row of each column.
const TITLE_TERMS: usize = 8;
const BODY_TERMS: usize = 60;
const TAGS_TERMS: usize = 3;
const COLUMNS: [&str; 3] = ["title", "body", "tags"];
const KS: [usize; 2] = [10, 100];
const SEED: u64 = 0x5eed_f1e1d5;
const TABLE: &str = "docs";
const NS_PER_SEC: f64 = 1e9;
const P50: usize = 50;
const P95: usize = 95;
/// Scores that differ by less than this share are the same score.
const SCORE_TOLERANCE: f32 = 1e-4;

/// The term of Zipf rank `rank`.
fn term(rank: usize) -> String {
    format!("w{rank:05}")
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(
        COLUMNS
            .iter()
            .map(|c| Field::new(*c, DataType::LargeUtf8, false))
            .collect::<Vec<_>>(),
    ))
}

/// `rows` rows of the three columns.
fn batch(
    schema: &SchemaRef,
    rows: usize,
    zipf: &ZipfDistribution,
    rng: &mut StdRng,
) -> RecordBatch {
    let mut column = |terms: usize| -> Arc<dyn Array> {
        Arc::new(LargeStringArray::from_iter_values((0..rows).map(|_| {
            (0..terms)
                .map(|_| term(zipf.sample(rng)))
                .collect::<Vec<_>>()
                .join(" ")
        })))
    };
    let arrays = vec![column(TITLE_TERMS), column(BODY_TERMS), column(TAGS_TERMS)];
    RecordBatch::try_new(Arc::clone(schema), arrays).expect("batch")
}

/// The queries timed, by label: clauses over every column.
fn queries() -> Vec<(String, Vec<FieldClause>)> {
    let clause = |ranks: &[usize]| {
        FieldClause::new(
            ranks.iter().map(|r| term(*r)).collect::<Vec<_>>().join(" "),
            COLUMNS,
        )
    };
    vec![
        ("one rare term".into(), vec![clause(&[5_000])]),
        ("two mid terms".into(), vec![clause(&[200, 900])]),
        ("one common term".into(), vec![clause(&[3])]),
        (
            "two clauses, mid + rare".into(),
            vec![clause(&[300]), clause(&[4_000])],
        ),
    ]
}

/// `(_id, score)` of every row of `batches`.
fn ids_scores(batches: &[RecordBatch]) -> Vec<(i128, f32)> {
    let mut out = Vec::new();
    for b in batches {
        let ids = b
            .column_by_name("_id")
            .expect("_id")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128")
            .clone();
        let scores = b
            .column_by_name("score")
            .expect("score")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32")
            .clone();
        out.extend((0..b.num_rows()).map(|i| (ids.value(i), scores.value(i))));
    }
    out
}

/// The best-fields top `k` of `clauses`, from single-column searches each
/// fetching `per_pair` rows (all of a pair's matches when `None`).
fn client_merge(
    table: &Supertable,
    clauses: &[FieldClause],
    k: usize,
    per_pair: Option<usize>,
    rows: usize,
) -> Vec<(i128, f32)> {
    let mut total: HashMap<i128, f32> = HashMap::new();
    for clause in clauses {
        let mut best: HashMap<i128, f32> = HashMap::new();
        for column in clause.fields() {
            let hits = table
                .bm25_search(
                    column,
                    clause.query(),
                    per_pair.unwrap_or(rows),
                    Bm25SearchOptions::new().with_mode(BoolMode::Or),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            for (id, score) in ids_scores(&hits) {
                let slot = best.entry(id).or_insert(0.0);
                *slot = slot.max(score);
            }
        }
        for (id, score) in best {
            *total.entry(id).or_insert(0.0) += score;
        }
    }
    let mut out: Vec<(i128, f32)> = total.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out
}

/// Whether two ranked lists carry the same scores, position by position.
/// Rows tied at the cut can differ, so ids are not compared.
fn same_scores(a: &[(i128, f32)], b: &[(i128, f32)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.1 - y.1).abs() <= SCORE_TOLERANCE * x.1.abs().max(1.0))
}

/// Share of `exact`'s rows `approx` also returns.
fn overlap(exact: &[(i128, f32)], approx: &[(i128, f32)]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let ids: std::collections::HashSet<i128> = approx.iter().map(|r| r.0).collect();
    exact.iter().filter(|r| ids.contains(&r.0)).count() as f64 / exact.len() as f64
}

/// Median and p95 of `iters` warm runs of `run`, after one untimed run.
fn time(iters: usize, mut run: impl FnMut()) -> (f64, f64) {
    run();
    let mut samples: Vec<Duration> = (0..iters)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect();
    let ns = |d: Duration| d.as_secs_f64() * NS_PER_SEC;
    (
        ns(percentile(&mut samples, P50)),
        ns(percentile(&mut samples, P95)),
    )
}

fn timing(p50: f64, p95: f64) -> [Cell; 2] {
    [
        metric(p50, fmt_time(p50), Better::Lower),
        metric(p95, fmt_time(p95), Better::Lower),
    ]
}

/// Run the diagnostic.
pub fn run() {
    let iters = env_usize("INFINO_DIAG_ITERS", DEFAULT_ITERS).max(1);
    let rows = env_usize("INFINO_BENCH_FTS_FIELDS_ROWS", DEFAULT_ROWS).max(ROWS_PER_COMMIT);
    let commits = rows / ROWS_PER_COMMIT;
    let rows = commits * ROWS_PER_COMMIT;
    let dir = TempDir::new().expect("table dir");
    let uri = dir.path().to_str().expect("utf8 path").to_string();
    let schema = schema();
    let zipf = ZipfDistribution::new(VOCAB);
    let mut rng = StdRng::seed_from_u64(SEED);

    eprintln!("[fts-fields] writing {commits} commits of {ROWS_PER_COMMIT} rows...");
    let conn = connect(&uri).expect("connect");
    let spec = COLUMNS
        .iter()
        .fold(IndexSpec::new(), |spec, c| spec.fts(*c));
    let table = conn
        .create_table(TABLE, Arc::clone(&schema), spec)
        .expect("create");
    for _ in 0..commits {
        table
            .append(&batch(&schema, ROWS_PER_COMMIT, &zipf, &mut rng))
            .expect("append");
    }
    let table = conn.open_table(TABLE).expect("open");
    let opts = Bm25FieldsOptions::new();

    let mut report_rows = Vec::new();
    for (label, clauses) in queries() {
        for k in KS {
            let fields = ids_scores(
                &table
                    .bm25_search_fields(&clauses, k, &opts, Some(&["_id", "score"]))
                    .expect("fields"),
            );
            let exact = client_merge(&table, &clauses, k, None, rows);
            assert!(
                same_scores(&fields, &exact),
                "{label} k={k}: bm25_search_fields differs from the exact client merge"
            );
            let approx = client_merge(&table, &clauses, k, Some(k), rows);
            let recall = overlap(&exact, &approx);

            let (f50, f95) = time(iters, || {
                black_box(
                    table
                        .bm25_search_fields(&clauses, k, &opts, Some(&["_id", "score"]))
                        .expect("fields"),
                );
            });
            let (e50, e95) = time(iters, || {
                black_box(client_merge(&table, &clauses, k, None, rows));
            });
            let (a50, a95) = time(iters, || {
                black_box(client_merge(&table, &clauses, k, Some(k), rows));
            });
            eprintln!(
                "[fts-fields] {label} k={k}: fields p50 {} | client every match p50 {} | \
                 client top-k p50 {} (overlap {recall:.3})",
                fmt_time(f50),
                fmt_time(e50),
                fmt_time(a50),
            );
            let mut row = vec![text(label.clone()), text(k.to_string())];
            row.extend(timing(f50, f95));
            row.extend(timing(e50, e95));
            row.extend(timing(a50, a95));
            row.push(metric(recall, format!("{recall:.3}"), Better::Higher));
            report_rows.push(row);
        }
    }

    let mut report = Report::load("fts-fields");
    report.emit(&Section {
        anchor: "bench/diagnostic/fts-fields".into(),
        title: "Multi-field BM25 against single-column searches merged in the client".into(),
        note: format!(
            "{rows} rows in {commits} commits; full-text columns title ({TITLE_TERMS} terms), \
             body ({BODY_TERMS}) and tags ({TAGS_TERMS}) over a {VOCAB}-term Zipfian \
             vocabulary; every clause searches all three. Warm median and p95 of {iters} runs. \
             `client, every match` fetches each (clause, column) pair's every match and is \
             exact; `client, top k` fetches each pair's top k, and `overlap` is the share of \
             the exact top k it returns."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: [
                "Query",
                "k",
                "fields p50",
                "fields p95",
                "client, every match p50",
                "client, every match p95",
                "client, top k p50",
                "client, top k p95",
                "overlap",
            ]
            .map(String::from)
            .to_vec(),
            rows: report_rows,
        }],
    });
    report.save();
}
