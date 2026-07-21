// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared harness for the query-path diagnostics (`sql-diag`,
//! `fts-diag`).
//!
//! Both diagnostics measure per-query paths over the *same* kind of
//! table, so they build and configure it the same way — one scale knob
//! (`INFINO_BENCH_SUPERTABLE_DOCS`), one iters knob (`INFINO_DIAG_ITERS`),
//! one corpus, one supertable build, one set of stat/format helpers.
//! Each diagnostic then plugs in only its own measurements. Without this
//! the diagnostics drift into separate config surfaces (different doc-count
//! knobs, different corpora, different stores), which is exactly what this
//! module exists to prevent.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_tokenizer,
};

use crate::corpus::{self, MmapTextCorpus};

/// Rows per commit — matches the headline benches' `WRITE_CHUNK` so the
/// diagnostic's superfile count mirrors production shapes.
pub const WRITE_CHUNK: usize = 65_536;

/// Round-robin category labels (matches `superfile::sql::CATEGORIES`).
pub const CATEGORIES: &[&str] = &["rust", "python", "go", "sql"];

/// Default timed iters per path; override with `INFINO_DIAG_ITERS`.
const DEFAULT_ITERS: usize = 15;

/// Unified diagnostic config: one scale knob, one iters knob, shared by
/// every query-path diagnostic.
pub struct DiagConfig {
    pub n_docs: usize,
    pub iters: usize,
}

/// Read the diagnostic config from the single shared set of knobs.
pub fn config() -> DiagConfig {
    let iters = std::env::var("INFINO_DIAG_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    DiagConfig {
        n_docs: corpus::supertable_docs(),
        iters,
    }
}

/// Scalar schema for the diagnostic table: `title` (FTS) + `category` +
/// `rating`. No vector index — vectors never touch the scored/scalar
/// paths these diagnostics measure, so omitting them keeps the build
/// cheap while leaving the measured paths identical.
pub fn diag_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("category", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
    ]))
}

/// Supertable options for the diagnostic table (scalar + FTS on `title`).
pub fn diag_options() -> SupertableOptions {
    SupertableOptions::new(
        diag_schema(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("diag supertable options")
}

/// Build one `WRITE_CHUNK`-row batch from `(id, text)` corpus rows.
pub fn chunk_batch(rows: &[(u64, &str)]) -> RecordBatch {
    let titles = LargeStringArray::from(rows.iter().map(|&(_, t)| t).collect::<Vec<_>>());
    let categories = LargeStringArray::from(
        rows.iter()
            .map(|&(id, _)| CATEGORIES[(id as usize) % CATEGORIES.len()])
            .collect::<Vec<_>>(),
    );
    let ratings = Int64Array::from(
        rows.iter()
            .map(|&(id, _)| (id % 100) as i64)
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(
        diag_schema(),
        vec![Arc::new(titles), Arc::new(categories), Arc::new(ratings)],
    )
    .expect("chunk batch")
}

/// Build the diagnostic `Supertable` from a Zipfian `termNNNNN` corpus,
/// committed one `WRITE_CHUNK` superfile at a time (in-memory store, warm
/// by construction). Returns the table plus the chunk batches — the SQL
/// diagnostic reuses them for its DataFusion baselines; callers that only
/// need the table can ignore them (the batches own their data, so the
/// corpus is dropped here).
pub fn build_supertable(cfg: &DiagConfig) -> (Supertable, Vec<RecordBatch>) {
    let corpus = MmapTextCorpus::generate(cfg.n_docs, 1);
    let batches: Vec<RecordBatch> = corpus.rows().chunks(WRITE_CHUNK).map(chunk_batch).collect();
    let table = Supertable::create(diag_options()).expect("create diag supertable");
    {
        let mut writer = table.writer().expect("writer");
        for batch in &batches {
            writer.append(batch).expect("append");
            writer.commit().expect("commit");
        }
    }
    (table, batches)
}

/// The `p`-th percentile of `samples` (sorts in place). `Duration::ZERO`
/// for an empty slice.
pub fn percentile(samples: &mut [Duration], p: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = ((p as f64 / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

/// Right-aligned µs/ms formatting, shared across diagnostic tables.
pub fn fmt(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 {
        format!("{us:>9.1} µs")
    } else {
        format!("{:>9.2} ms", us / 1000.0)
    }
}

/// Time `f` once (warm-up) then `iters` times; return (p50, mean, rows).
pub fn time_path(iters: usize, mut f: impl FnMut() -> usize) -> (Duration, Duration, usize) {
    let rows = f();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let out = f();
        samples.push(t.elapsed());
        std::hint::black_box(out);
    }
    let sum: u128 = samples.iter().map(|d| d.as_nanos()).sum();
    let mean = Duration::from_nanos((sum / samples.len().max(1) as u128) as u64);
    (percentile(&mut samples, 50), mean, rows)
}
