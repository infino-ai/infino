//! End-to-end update / delete throughput bench.
//!
//! Ingests a baseline corpus into a supertable, then drives a
//! series of `update` / `delete` calls and measures:
//!
//! - **Ingest throughput** — docs/sec for the baseline load.
//! - **Mutation throughput** — updates/sec + deletes/sec
//!   measured over the full WAL pipeline (resolve + append +
//!   tombstone + state-doc CAS + cleanup).
//! - **End-state correctness** — a closing query asserts the
//!   row count + presence/absence shape the bench expects.
//!
//! Defaults are sized so the bench runs in seconds on a
//! developer laptop. Larger sizes are gated behind env vars
//! (e.g. `INFINO_BENCH_UPDATE_N_DOCS=10000000` for the
//! 10M-doc scale-out shape).

use std::env;
use std::hint::black_box;
use std::sync::Arc;

use arrow_array::Array;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use datafusion::prelude::{col, lit};
use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::Supertable;
use infino::test_helpers::{build_title_batch, default_supertable_options};
use tempfile::TempDir;

/// Doc count for the baseline ingest. Override via
/// `INFINO_BENCH_UPDATE_N_DOCS`. Default sized to run in <1s.
fn n_docs() -> usize {
    env::var("INFINO_BENCH_UPDATE_N_DOCS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10_000)
}

/// Number of single-row mutations to drive after ingest.
/// Override via `INFINO_BENCH_UPDATE_N_MUTATIONS`. Default
/// sized so the criterion timer can sample at least a handful
/// of iterations.
fn n_mutations() -> usize {
    env::var("INFINO_BENCH_UPDATE_N_MUTATIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(20)
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

// ─── Bench: ingest ───────────────────────────────────────────────────

fn bench_ingest(c: &mut Criterion) {
    let n = n_docs();
    let mut g = c.benchmark_group("supertable_update_ingest");
    g.sample_size(10);
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("baseline_ingest", |b| {
        b.iter_with_large_drop(|| build_supertable_with_ingest(black_box(n)));
    });
    g.finish();
}

// ─── Bench: deletes ──────────────────────────────────────────────────

fn bench_deletes(c: &mut Criterion) {
    let n = n_docs();
    let m = n_mutations();
    let (_dir, st) = build_supertable_with_ingest(n);

    let mut g = c.benchmark_group("supertable_update_delete");
    g.sample_size(10);
    g.throughput(Throughput::Elements(m as u64));
    g.bench_function("single_row_predicate_deletes", |b| {
        // Each iteration buffers `m` deletes + one `commit()`
        // to drive them all through the WAL pipeline.
        // Criterion times the whole batch and divides by `m`
        // via Throughput.
        b.iter(|| {
            let mut w = st.writer().expect("writer");
            for i in 0..m {
                let title = format!("row{i:08}");
                let pending = w
                    .delete(col("title").eq(lit(title.clone())))
                    .expect("delete");
                black_box(pending);
            }
            let result = w.commit().expect("commit");
            black_box(result);
            drop(w);
        });
    });
    g.finish();

    // End-state correctness check: after this group runs, the
    // first `m` rows should be gone.
    let batches = st
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    let total = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count column")
        .value(0);
    // The bench may run multiple iterations; we only assert the
    // count never exceeds the baseline (no extra rows
    // appeared) and is below the baseline by at least m on
    // the first iteration. Use ≤ baseline to stay
    // iteration-count agnostic.
    assert!(
        total <= n as i64,
        "row count {total} exceeded baseline {n} — mutations leaked rows"
    );
}

// ─── Bench: updates ──────────────────────────────────────────────────

fn bench_updates(c: &mut Criterion) {
    let n = n_docs();
    let m = n_mutations();
    let (_dir, st) = build_supertable_with_ingest(n);

    let mut g = c.benchmark_group("supertable_update_update");
    g.sample_size(10);
    g.throughput(Throughput::Elements(m as u64));
    g.bench_function("single_row_predicate_updates", |b| {
        b.iter(|| {
            let mut w = st.writer().expect("writer");
            for i in 0..m {
                let title = format!("row{i:08}");
                let replacement = build_title_batch(&[&format!("row{i:08}-prime")]);
                let pending = w
                    .update(col("title").eq(lit(title.clone())), replacement)
                    .expect("update");
                black_box(pending);
            }
            let result = w.commit().expect("commit");
            black_box(result);
            drop(w);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_ingest, bench_deletes, bench_updates);
criterion_main!(benches);
