// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! What `optimize()` costs on a freshly built small hybrid table.
//!
//! A hybrid table (one BM25 column, one vector column) of a few thousand
//! rows is built with a single append and optimized once. That optimize
//! drains the rows into the hidden vector-cell index and recalibrates its
//! probe laws, so its time is dominated by the calibration's per-cell work:
//! small tables split into a couple of hundred cells of a few dozen rows
//! each, and any cost paid per cell rather than per row shows up here first.
//!
//! Reports the median optimize wall time per table shape, over fresh tables.
//!
//! Invoked as `cargo bench -- hybrid-optimize`.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{IndexSpec, Metric, OptimizeOptions, connect};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

use crate::{
    diag_common::{env_usize, percentile},
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
};

/// `(rows, vector dim)` per measured table.
const SHAPES: [(usize, usize); 4] = [(600, 64), (2_400, 64), (6_000, 64), (6_000, 384)];
/// Fresh tables built and optimized per shape; override with
/// `INFINO_DIAG_ITERS`.
const DEFAULT_ITERS: usize = 3;
/// Vector clusters the rows are drawn around.
const CLUSTERS: usize = 32;
/// Each row's offset from its cluster centre, per component, is uniform in
/// `±SPREAD / 2`.
const SPREAD: f32 = 0.6;
/// Words per row in the text column.
const WORDS_PER_ROW: usize = 12;
/// Fixed seed so every run builds the same tables.
const SEED: u64 = 7;
/// The text column's vocabulary.
const VOCAB: [&str; 24] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega",
];
const NS_PER_SEC: f64 = 1e9;
/// Median.
const P50: usize = 50;

fn schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("text", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

/// `rows` rows of random text and clustered `dim`-wide vectors.
fn batch(rows: usize, dim: usize) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(SEED);
    let centres: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let mut texts = Vec::with_capacity(rows);
    let mut flat = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        let words: Vec<&str> = (0..WORDS_PER_ROW)
            .map(|_| VOCAB[rng.random_range(0..VOCAB.len())])
            .collect();
        texts.push(words.join(" "));
        let centre = &centres[rng.random_range(0..CLUSTERS)];
        flat.extend(
            centre
                .iter()
                .map(|x| x + (rng.random::<f32>() - 0.5) * SPREAD),
        );
    }
    let emb = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
        None,
    )
    .expect("vector column");
    RecordBatch::try_new(
        schema(dim),
        vec![Arc::new(LargeStringArray::from(texts)), Arc::new(emb)],
    )
    .expect("batch")
}

/// Build a fresh hybrid table from `rows` and time one default optimize.
fn optimize_once(rows: &RecordBatch, dim: usize) -> Duration {
    let dir = TempDir::new().expect("table dir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let table = db
        .create_table(
            "t",
            schema(dim),
            IndexSpec::new()
                .fts("text")
                .vector("emb", dim, Metric::Cosine),
        )
        .expect("create");
    table.append(rows).expect("append");
    let started = Instant::now();
    table
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    started.elapsed()
}

/// Run the diagnostic.
pub fn run() {
    let iters = env_usize("INFINO_DIAG_ITERS", DEFAULT_ITERS).max(1);
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    for (n_rows, dim) in SHAPES {
        let data = batch(n_rows, dim);
        let mut times: Vec<Duration> = (0..iters).map(|_| optimize_once(&data, dim)).collect();
        let p50 = percentile(&mut times, P50).as_secs_f64() * NS_PER_SEC;
        eprintln!(
            "[hybrid-optimize] {n_rows} rows x {dim}d: optimize p50 {}",
            fmt_time(p50)
        );
        rows.push(vec![
            text(fmt_count(n_rows)),
            text(dim.to_string()),
            metric(p50, fmt_time(p50), Better::Lower),
        ]);
    }

    let mut report = Report::load("hybrid-optimize");
    report.emit(&Section {
        anchor: "bench/diagnostic/hybrid-optimize".into(),
        title: "Optimize after building a small hybrid table".into(),
        note: format!(
            "One append of N rows (a {WORDS_PER_ROW}-word BM25 column and a cosine vector column \
             clustered around {CLUSTERS} centres), then one default `optimize()`, which drains \
             the rows into the hidden vector-cell index and recalibrates its probe laws. Median \
             of {iters} fresh tables per shape, local filesystem."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec!["Rows".into(), "Dim".into(), "Optimize p50".into()],
            rows,
        }],
    });
    report.save();
}
