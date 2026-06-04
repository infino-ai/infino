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

/// Total live row count via SQL `COUNT(*)`.
fn count_rows(st: &Supertable) -> i64 {
    let batches = st
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count column")
        .value(0)
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
    // `i` lives outside the bench closure on purpose: criterion calls
    // that closure several times (estimate, warm-up, measurement), and
    // the deletes mutate the one reused supertable. A counter reset to
    // 0 on a later invocation would re-target already-tombstoned rows
    // and silently no-op, so it must run monotonically across them.
    let mut i: u64 = 0;
    g.bench_function("single_row_predicate_deletes", |b| {
        // Delete distinct rows by predicate, one `commit()` per delete,
        // against a single reused supertable — no per-iteration
        // rebuild. Each delete targets a fresh, still-present row
        // (`row{i:08}`). Unlike the update chain a delete is terminal,
        // so this consumes the corpus: the `n`-row pool bounds how many
        // deletes a run can make before predicates start matching
        // nothing (ample at the scale-out sizes).
        b.iter(|| {
            for _ in 0..m {
                let mut w = st.writer().expect("writer");
                let title = format!("row{i:08}");
                let pending = w.delete(col("title").eq(lit(title))).expect("delete");
                black_box(pending);
                black_box(w.commit().expect("commit"));
                i += 1;
            }
        });
    });
    g.finish();

    // Exact end-state: each of the `i` delete attempts removed a
    // distinct row while one was still present, so exactly
    // `min(i, n)` rows are gone and `n - min(i, n)` remain. Asserting
    // the exact count (not just `< n`) catches both under- and
    // over-deletion — e.g. a predicate that silently matched nothing
    // or matched more than one row.
    let expected = (n as u64).saturating_sub(i) as i64;
    assert_eq!(
        count_rows(&st),
        expected,
        "after {i} single-row deletes over an {n}-row table, \
         expected {expected} rows to remain"
    );
}

// ─── Bench: updates ──────────────────────────────────────────────────

fn bench_updates(c: &mut Criterion) {
    let n = n_docs();
    let m = n_mutations();
    let (_dir, st) = build_supertable_with_ingest(n);

    // Seed the single row this bench rewrites, at generation 0. It
    // sits alongside the `n`-row corpus, so each update's predicate
    // still resolves against a realistically-sized table.
    {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["target-0"])).expect("append");
        w.commit().expect("commit");
    }

    let mut g = c.benchmark_group("supertable_update_update");
    g.sample_size(10);
    g.throughput(Throughput::Elements(m as u64));
    // `i` lives outside the bench closure on purpose: criterion calls
    // that closure several times (estimate, warm-up, measurement), and
    // the chain's state lives in the one reused supertable. A counter
    // reset to 0 on a later invocation would look for `target-0` —
    // long since rewritten to `target-{N}` — and fail the cardinality
    // contract with `matched: 0`. It must run monotonically across all
    // invocations.
    let mut i: u64 = 0;
    g.bench_function("single_row_predicate_updates", |b| {
        // Rewrite one row over and over, advancing a generation suffix:
        // `target-{i}` -> `target-{i+1}`. Each step must `commit()`
        // before the next, because `update` resolves its predicate
        // against the *committed* manifest snapshot — the previous
        // rename has to be durable for the next predicate to match.
        // The same supertable is reused throughout — no per-iteration
        // rebuild.
        b.iter(|| {
            for _ in 0..m {
                let mut w = st.writer().expect("writer");
                let from = format!("target-{i}");
                let to = format!("target-{}", i + 1);
                let replacement = build_title_batch(&[&to]);
                let pending = w
                    .update(col("title").eq(lit(from)), replacement)
                    .expect("update");
                black_box(pending);
                black_box(w.commit().expect("commit"));
                i += 1;
            }
        });
    });
    g.finish();

    // Update preserves the row count: the `n`-row corpus plus the
    // single rewritten target row.
    assert_eq!(
        count_rows(&st),
        (n + 1) as i64,
        "update changed the row count; expected {}",
        n + 1
    );
}

criterion_group!(benches, bench_ingest, bench_deletes, bench_updates);
criterion_main!(benches);
