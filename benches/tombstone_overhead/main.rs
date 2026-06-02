//! Hot-path overhead of the reader-side tombstone filter.
//!
//! Measures per-query latency across three supertable states so a
//! regression in the cache + filter hot path is localized to one
//! of: DashMap lookup cost, TTL-check cost, filter-hook cost, or
//! the cache-miss path.
//!
//! ## States
//!
//! - **`clean`** — no tombstones ever written. The cache fills
//!   with "known 404" sentinels on first lookup; every subsequent
//!   query hits the `bitmap.is_empty()` short-circuit. This is
//!   the steady-state floor a normally-operating supertable
//!   should sit at.
//!
//! - **`one_percent`** — 1 % of docs tombstoned, distributed
//!   evenly across superfiles. Roughly half of per-superfile
//!   lookups still short-circuit on empty; the other half iterate
//!   a small Roaring bitmap once per query.
//!
//! - **`ten_percent_churned`** — 10 % tombstoned, then the same
//!   rows re-tombstoned to exercise the writer's CAS-loss /
//!   bitmap-union path AND ensure the cache's
//!   `bitmap.is_empty()` short-circuit is NOT taken on the
//!   read path. Stresses the full filter loop.
//!
//! ## Diff against a baseline
//!
//! Criterion saves runs to `target/criterion/`. To compare this
//! branch's overhead to `main`:
//!
//! ```text
//! git checkout main
//! cargo bench --bench tombstone-overhead
//! cp -r target/criterion target/criterion-main-baseline
//! git checkout <this branch>
//! cargo bench --bench tombstone-overhead
//! # Compare `target/criterion-main-baseline/<group>` to
//! # `target/criterion/<group>` per sub-bench (estimates.json
//! # in each carries the p50 / p95 in ns).
//! ```
//!
//! Target per-sub-bench budget: `delta ≤ max(2 % × base_p50,
//! 1 µs)`. Localize a regression by group: a `clean` regression
//! points at lookup / TTL costs; a `one_percent` regression
//! points at the filter-hook path; a `ten_percent_churned`
//! regression points at filter cost on tombstone-heavy
//! superfiles.

use std::hint::black_box;
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use chrono::Utc;
use criterion::{Criterion, criterion_group, criterion_main};
use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::superfile::fts::reader::BoolMode;
use infino::supertable::Supertable;
use infino::supertable::wal::WalStore;
use infino::supertable::wal::pipeline::run_tombstone_phase;
use infino::supertable::wal::state_doc::{
    OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState, WalStateDoc,
};
use infino::test_helpers::{build_title_batch, default_supertable_options};
use tempfile::TempDir;

// ─── Sizing ───────────────────────────────────────────────────────────

/// Doc count. Sized down from the 10M-doc FTS bench so the
/// overhead bench runs in seconds even with the tombstone and
/// WAL drive-around per state. Large enough that per-query
/// overhead at the filter hook is measurable above the noise
/// floor.
const N_DOCS: usize = 50_000;

/// Append-chunk count. Each chunk becomes one row-shard which
/// the writer turns into one superfile. The bench's "manifest
/// shape" is then 8 superfiles, which gives enough fan-out for
/// the per-superfile filter overhead to dominate over the
/// orchestrator's fixed costs.
const APPEND_CHUNKS: usize = 8;

/// Top-K for the search query — sized to be representative of a
/// real query workload's top-of-list shape.
const TOP_K: usize = 10;

/// One BM25 search per query. Picked to hit roughly half the
/// superfiles in pruning (varies with the corpus); the cache
/// hook fires once per touched superfile.
const QUERY_TERM: &str = "alpha";

// ─── Fixtures ─────────────────────────────────────────────────────────

/// Build one of the three workload supertables. Each variant
/// uses its own `TempDir` so storage state stays isolated.
fn build_supertable(state: WorkloadState) -> (TempDir, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    // Append + commit. The corpus is synthetic: each row carries
    // the search-term plus a unique disambiguator so FTS hits
    // every row but BM25 ordering varies.
    let mut w = st.writer().expect("writer");
    let chunk_size = N_DOCS.div_ceil(APPEND_CHUNKS);
    for chunk_idx in 0..APPEND_CHUNKS {
        let start = chunk_idx * chunk_size;
        let end = ((chunk_idx + 1) * chunk_size).min(N_DOCS);
        if start >= end {
            break;
        }
        let titles_owned: Vec<String> = (start..end).map(|i| format!("alpha row{i:08}")).collect();
        let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
        let batch: RecordBatch = build_title_batch(&titles);
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    // Drive tombstones for the non-clean variants.
    let ws = WalStore::new(Arc::clone(&storage));
    match state {
        WorkloadState::Clean => {}
        WorkloadState::OnePercent => {
            drive_tombstones(&st, &ws, 0.01, false);
        }
        WorkloadState::TenPercentChurned => {
            drive_tombstones(&st, &ws, 0.10, true);
        }
    }

    (dir, st)
}

#[derive(Debug, Clone, Copy)]
enum WorkloadState {
    Clean,
    OnePercent,
    TenPercentChurned,
}

/// Tombstone the first `fraction` of each superfile's docs.
/// `churn` re-runs the same WAL pipeline a second time so the
/// sidecar bitmap is hit twice (idempotent union; bitmap stays
/// the same shape but the cache's last-written etag advances).
fn drive_tombstones(st: &Supertable, ws: &WalStore, fraction: f64, churn: bool) {
    let manifest = st.reader().manifest().clone();
    let mut targets: Vec<i128> = Vec::new();
    for entry in manifest.superfile_list.superfiles.iter() {
        let n = (entry.n_docs as f64 * fraction).ceil() as i64;
        for i in 0..n {
            targets.push(entry.id_min + i as i128);
        }
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .build()
        .expect("rt");
    rt.block_on(async move {
        let wal_id_base: i128 = 100_000_000;
        for (i, &target) in targets.iter().enumerate() {
            let wal = build_delete_wal(target, wal_id_base + i as i128);
            let etag = ws.create(&wal).await.expect("wal create");
            run_tombstone_phase(st, ws, &wal, &etag)
                .await
                .expect("tombstone phase");
            if churn {
                let churn_wal = build_delete_wal(target, wal_id_base * 2 + i as i128);
                let churn_etag = ws.create(&churn_wal).await.expect("wal create");
                run_tombstone_phase(st, ws, &churn_wal, &churn_etag)
                    .await
                    .expect("tombstone phase churn");
            }
        }
    });
}

fn build_delete_wal(target_id: i128, wal_id_value: i128) -> WalStateDoc {
    WalStateDoc {
        wal_id: WalId(wal_id_value),
        schema_version: SCHEMA_VERSION,
        op_kind: OpKind::Delete,
        state: WalState::Intent,
        created_at: Utc::now(),
        lease: None,
        predicate_repr: "bench".into(),
        target_ids: vec![RowId(target_id)],
        new_row_count: None,
        new_row_content_hash: None,
        preallocated_superfile_id: None,
        minted_id_spans: Vec::new(),
        tombstone_progress: vec![TombstoneEntry {
            target_id: RowId(target_id),
            outcome: TombstoneOutcome::Pending,
            tombstoned_in_superfile: None,
        }],
    }
}

// Cached fixtures so each criterion sample reuses one set.
fn fixture_clean() -> &'static Supertable {
    static F: OnceLock<(TempDir, Supertable)> = OnceLock::new();
    &F.get_or_init(|| build_supertable(WorkloadState::Clean)).1
}

fn fixture_one_percent() -> &'static Supertable {
    static F: OnceLock<(TempDir, Supertable)> = OnceLock::new();
    &F.get_or_init(|| build_supertable(WorkloadState::OnePercent))
        .1
}

fn fixture_ten_percent_churned() -> &'static Supertable {
    static F: OnceLock<(TempDir, Supertable)> = OnceLock::new();
    &F.get_or_init(|| build_supertable(WorkloadState::TenPercentChurned))
        .1
}

// ─── Benches ──────────────────────────────────────────────────────────

fn bench_fts(c: &mut Criterion) {
    let mut g = c.benchmark_group("tombstone_overhead_fts");
    g.sample_size(20);

    g.bench_function("clean", |b| {
        let st = fixture_clean();
        b.iter(|| {
            let hits = st
                .bm25_search(
                    black_box("title"),
                    black_box(QUERY_TERM),
                    black_box(TOP_K),
                    BoolMode::Or,
                )
                .expect("fts");
            black_box(hits);
        });
    });

    g.bench_function("one_percent", |b| {
        let st = fixture_one_percent();
        b.iter(|| {
            let hits = st
                .bm25_search(
                    black_box("title"),
                    black_box(QUERY_TERM),
                    black_box(TOP_K),
                    BoolMode::Or,
                )
                .expect("fts");
            black_box(hits);
        });
    });

    g.bench_function("ten_percent_churned", |b| {
        let st = fixture_ten_percent_churned();
        b.iter(|| {
            let hits = st
                .bm25_search(
                    black_box("title"),
                    black_box(QUERY_TERM),
                    black_box(TOP_K),
                    BoolMode::Or,
                )
                .expect("fts");
            black_box(hits);
        });
    });

    g.finish();
}

fn bench_sql(c: &mut Criterion) {
    let mut g = c.benchmark_group("tombstone_overhead_sql");
    g.sample_size(20);

    g.bench_function("clean", |b| {
        let st = fixture_clean();
        b.iter(|| {
            let batches = st
                .query_sql(black_box("SELECT COUNT(*) FROM supertable"))
                .expect("sql");
            black_box(batches);
        });
    });

    g.bench_function("one_percent", |b| {
        let st = fixture_one_percent();
        b.iter(|| {
            let batches = st
                .query_sql(black_box("SELECT COUNT(*) FROM supertable"))
                .expect("sql");
            black_box(batches);
        });
    });

    g.bench_function("ten_percent_churned", |b| {
        let st = fixture_ten_percent_churned();
        b.iter(|| {
            let batches = st
                .query_sql(black_box("SELECT COUNT(*) FROM supertable"))
                .expect("sql");
            black_box(batches);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_fts, bench_sql);
criterion_main!(benches);
