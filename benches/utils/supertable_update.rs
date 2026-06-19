// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end update / delete throughput bench.
//!
//! Ingests a baseline corpus into a supertable, then drives a
//! series of `update` / `delete` calls and measures:
//!
//! - **Ingest throughput** — docs/sec for the baseline load.
//! - **Mutation throughput** — updates/sec + deletes/sec
//!   measured over the full WAL pipeline (resolve + append +
//!   tombstone + state-doc CAS + cleanup).
//! - **End-state correctness** — a closing assertion checks the
//!   row-count invariant each mutation implies: updates preserve
//!   the count, deletes shrink it.
//!
//! Defaults are sized so the bench runs in seconds on a
//! developer laptop. Larger sizes are gated behind env vars
//! (e.g. `INFINO_BENCH_UPDATE_N_DOCS=10000000` for the
//! 10M-doc scale-out shape). Results render through the custom
//! report harness (terminal +, when `INFINO_BENCH_UPDATE_README=1`,
//! the `bench/supertable_update/*` README anchors) with run-to-run
//! deltas.
//!
//! Invoked as `cargo bench -- update`.

use std::{
    env,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use datafusion::prelude::{col, lit};
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::Supertable,
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

use crate::{
    markdown::{fmt_count, fmt_throughput, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::{self, PeakSampler, RssStats},
};

/// Default baseline ingest doc count (sized to run in <1s) when
/// `INFINO_BENCH_UPDATE_N_DOCS` is unset.
const DEFAULT_N_DOCS: usize = 10_000;
/// Default number of post-ingest single-row mutations when
/// `INFINO_BENCH_UPDATE_N_MUTATIONS` is unset.
const DEFAULT_N_MUTATIONS: usize = 20;
/// Nanoseconds per second, for latency markdown.
const NS_PER_SEC: f64 = 1e9;

/// Doc count for the baseline ingest. Override via
/// `INFINO_BENCH_UPDATE_N_DOCS`. Default sized to run in <1s.
fn n_docs() -> usize {
    env::var("INFINO_BENCH_UPDATE_N_DOCS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_N_DOCS)
}

/// Number of single-row mutations to drive after ingest.
/// Override via `INFINO_BENCH_UPDATE_N_MUTATIONS`. Default
/// sized so the timer can sample a handful of iterations.
fn n_mutations() -> usize {
    env::var("INFINO_BENCH_UPDATE_N_MUTATIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_N_MUTATIONS)
}

/// Build a supertable + ingest a corpus of `n` rows in one
/// commit. Each row's `title` is unique so per-row deletes
/// resolve to one row each.
fn build_supertable_with_ingest(n: usize) -> (TempDir, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let titles_owned: Vec<String> = (0..n).map(|i| format!("row{i:08}")).collect();
    let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&titles)).expect("append");
    w.commit().expect("commit");
    drop(w);
    (dir, st)
}

fn rss_cells(stats: RssStats) -> Vec<Cell> {
    vec![
        metric(
            stats.peak_rss_bytes as f64,
            rss::fmt_bytes(stats.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.median_rss_bytes as f64,
            rss::fmt_bytes(stats.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.p90_rss_bytes as f64,
            rss::fmt_bytes(stats.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

/// Total live row count via SQL `COUNT(*)`.
fn count_rows(st: &Supertable) -> i64 {
    let batches = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count column")
        .value(0)
}

// ─── Ingest ───────────────────────────────────────────────────────────

/// Time one baseline ingest of `n` rows; returns the wall time + RSS. The
/// built supertable is dropped before returning (large-drop excluded
/// from the timed span, matching the old `iter_with_large_drop`).
fn measure_ingest(n: usize) -> (Duration, RssStats) {
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = build_supertable_with_ingest(black_box(n));
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();
    drop(built);
    (wall, rss)
}

// ─── Deletes ──────────────────────────────────────────────────────────

/// Drive `m` distinct single-row deletes against a fresh `n`-row table,
/// one `commit()` per delete. Returns the wall time and asserts the
/// exact end-state row count. Each delete targets a fresh, still-present
/// row (`row{i:08}`); the counter runs monotonically.
fn measure_deletes(n: usize, m: usize) -> (Duration, RssStats) {
    let (_dir, st) = build_supertable_with_ingest(n);
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    for i in 0..m as u64 {
        let mut w = st.writer().expect("writer");
        let title = format!("row{i:08}");
        let pending = w.delete(col("title").eq(lit(title))).expect("delete");
        black_box(pending);
        black_box(w.commit().expect("commit"));
    }
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();

    // Exact end-state: each of the `m` delete attempts removed a
    // distinct row while one was still present, so exactly
    // `min(m, n)` rows are gone and `n - min(m, n)` remain.
    let expected = (n as u64).saturating_sub(m as u64) as i64;
    assert_eq!(
        count_rows(&st),
        expected,
        "after {m} single-row deletes over an {n}-row table, \
         expected {expected} rows to remain"
    );
    (wall, rss)
}

// ─── Updates ──────────────────────────────────────────────────────────

/// Rewrite one row `m` times (`target-{i}` -> `target-{i+1}`) against a
/// fresh `n`-row table, one `commit()` per update. Returns the wall time
/// and asserts the row count is preserved (`n + 1`).
fn measure_updates(n: usize, m: usize) -> (Duration, RssStats) {
    let (_dir, st) = build_supertable_with_ingest(n);

    // Seed the single row this bench rewrites, at generation 0. It
    // sits alongside the `n`-row corpus, so each update's predicate
    // still resolves against a realistically-sized table.
    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["target-0"])).expect("append");
        w.commit().expect("commit");
    }

    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    for i in 0..m as u64 {
        let mut w = st.writer().expect("writer");
        let from = format!("target-{i}");
        let to = format!("target-{}", i + 1);
        let replacement = build_title_batch(&[&to]);
        let pending = w
            .update(col("title").eq(lit(from)), replacement)
            .expect("update");
        black_box(pending);
        black_box(w.commit().expect("commit"));
    }
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();

    // Update preserves the row count: the `n`-row corpus plus the
    // single rewritten target row.
    assert_eq!(
        count_rows(&st),
        (n + 1) as i64,
        "update changed the row count; expected {}",
        n + 1
    );
    (wall, rss)
}

// ─── Entry point ──────────────────────────────────────────────────────

pub fn run() {
    let n = n_docs();
    let m = n_mutations();
    eprintln!(
        "[supertable-update] baseline {} docs, {} single-row mutations...",
        fmt_count(n),
        fmt_count(m)
    );

    let (ingest_wall, ingest_rss) = measure_ingest(n);
    let (delete_wall, delete_rss) = measure_deletes(n, m);
    let (update_wall, update_rss) = measure_updates(n, m);

    let ingest_ns = ingest_wall.as_secs_f64() * NS_PER_SEC;
    let ingest_thr = n as f64 / ingest_wall.as_secs_f64();

    let mutation_row = |label: &str, wall: Duration, rss: RssStats| -> Vec<Cell> {
        let ns = wall.as_secs_f64() * NS_PER_SEC;
        let ops = m as f64 / wall.as_secs_f64();
        let mut cells = vec![
            text(label),
            metric(ns, fmt_time(ns), Better::Lower),
            metric(ops, format!("{ops:.0}/s"), Better::Higher),
        ];
        cells.extend(rss_cells(rss));
        cells
    };

    let mut ingest_cells = vec![
        text("baseline_ingest"),
        metric(ingest_ns, fmt_time(ingest_ns), Better::Lower),
        metric(ingest_thr, fmt_throughput(ingest_thr), Better::Higher),
    ];
    ingest_cells.extend(rss_cells(ingest_rss));

    let mut report = Report::load("supertable-update");
    report.emit(&Section {
        anchor: "bench/supertable_update/ingest".into(),
        title: format!(
            "Supertable update — baseline ingest ({} docs)",
            fmt_count(n)
        ),
        note: "Single-commit baseline load via `SupertableWriter::append` + `commit`. \
               Throughput is rows/s. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Phase".into(),
                "Time".into(),
                "Throughput".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows: vec![ingest_cells],
        }],
    });
    report.emit(&Section {
        anchor: "bench/supertable_update/mutation".into(),
        title: format!(
            "Supertable update — mutation throughput ({} single-row ops over a {}-doc table)",
            fmt_count(m),
            fmt_count(n)
        ),
        note: "End-to-end single-row `update` / `delete`, one `commit()` each, through the full \
               WAL pipeline (resolve + append + tombstone + state-doc CAS + cleanup). End-state \
               row counts are asserted (deletes shrink the table, updates preserve it). Throughput \
               is mutations/s. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Op".into(),
                "Time".into(),
                "Throughput".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows: vec![
                mutation_row("single_row_predicate_deletes", delete_wall, delete_rss),
                mutation_row("single_row_predicate_updates", update_wall, update_rss),
            ],
        }],
    });
    report.save();
}
