// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Warm-path overhead of the reader-side tombstone filter.
//!
//! Measures per-query latency across three supertable states so a
//! regression in the cache + filter warm path is localized to one
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
//! Reports the per-state p50 for an FTS (BM25) query and a SQL
//! `COUNT(*)` query through the custom report harness (terminal +,
//! when `INFINO_BENCH_UPDATE_README=1`, the `bench/tombstone/overhead`
//! README anchor), with run-to-run deltas. A `clean` regression points
//! at lookup / TTL costs; a `one_percent` regression points at the
//! filter-hook path; a `ten_percent_churned` regression points at filter
//! cost on tombstone-heavy superfiles.
//!
//! Invoked as `cargo bench -- tombstone`.

use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::RecordBatch;
use chrono::Utc;
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        wal::{
            WalStore,
            pipeline::run_tombstone_phase,
            state_doc::{
                OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
                WalStateDoc,
            },
        },
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

use crate::{
    markdown::fmt_time,
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::{self, PeakSampler, RssStats},
};

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

/// Tombstone fraction for the 1%-deleted workload variant.
const ONE_PERCENT_TOMBSTONE_FRACTION: f64 = 0.01;
/// Tombstone fraction for the 10%-deleted (churned) workload variant.
const TEN_PERCENT_TOMBSTONE_FRACTION: f64 = 0.10;
/// Tokio worker threads for driving the WAL tombstone pipeline.
const WAL_RUNTIME_WORKER_THREADS: usize = 2;
/// Base synthetic WAL id, kept well clear of any real ingest ids.
const WAL_ID_BASE: i128 = 100_000_000;
/// Multiplier offsetting the second (churn) pass's WAL ids from the base.
const CHURN_WAL_ID_MULTIPLIER: i128 = 2;

/// Top-K for the search query — sized to be representative of a
/// real query workload's top-of-list shape.
const TOP_K: usize = 10;

/// One BM25 search per query. Picked to hit roughly half the
/// superfiles in pruning (varies with the corpus); the cache
/// hook fires once per touched superfile.
const QUERY_TERM: &str = "alpha";

/// Timed repetitions per query (after one warmup); report the p50.
/// Matches the Criterion `sample_size(20)` the bench previously used.
const ITERS: usize = 20;
/// Nanoseconds per second, for latency markdown.
const NS_PER_SEC: f64 = 1e9;

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
            drive_tombstones(&st, &ws, ONE_PERCENT_TOMBSTONE_FRACTION, false);
        }
        WorkloadState::TenPercentChurned => {
            drive_tombstones(&st, &ws, TEN_PERCENT_TOMBSTONE_FRACTION, true);
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

impl WorkloadState {
    fn label(self) -> &'static str {
        match self {
            WorkloadState::Clean => "clean",
            WorkloadState::OnePercent => "one_percent",
            WorkloadState::TenPercentChurned => "ten_percent_churned",
        }
    }
}

/// Tombstone the first `fraction` of each superfile's docs.
/// `churn` re-runs the same WAL pipeline a second time so the
/// sidecar bitmap is hit twice (idempotent union; bitmap stays
/// the same shape but the cache's last-written etag advances).
fn drive_tombstones(st: &Supertable, ws: &WalStore, fraction: f64, churn: bool) {
    let manifest = st.reader().manifest().clone();
    let mut targets: Vec<i128> = Vec::new();
    for entry in manifest.get_all_superfiles().iter() {
        let n = (entry.n_docs as f64 * fraction).ceil() as i64;
        for i in 0..n {
            targets.push(entry.id_min + i as i128);
        }
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(WAL_RUNTIME_WORKER_THREADS)
        .build()
        .expect("rt");
    rt.block_on(async move {
        let wal_id_base: i128 = WAL_ID_BASE;
        for (i, &target) in targets.iter().enumerate() {
            let wal = build_delete_wal(target, wal_id_base + i as i128);
            let etag = ws.create(&wal).await.expect("wal create");
            run_tombstone_phase(st, ws, &wal, &etag)
                .await
                .expect("tombstone phase");
            if churn {
                let churn_wal =
                    build_delete_wal(target, wal_id_base * CHURN_WAL_ID_MULTIPLIER + i as i128);
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

// ─── Measurement ──────────────────────────────────────────────────────

fn p50(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

/// p50 of the BM25 query over `ITERS` timed runs (after one warmup).
fn measure_fts(st: &Supertable) -> Duration {
    let warm = st
        .reader()
        .bm25_search("title", QUERY_TERM, TOP_K, BoolMode::Or, None)
        .expect("fts");
    black_box(warm);
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let hits = st
            .reader()
            .bm25_search(
                black_box("title"),
                black_box(QUERY_TERM),
                black_box(TOP_K),
                BoolMode::Or,
                None,
            )
            .expect("fts");
        samples.push(t0.elapsed());
        black_box(hits);
    }
    p50(&mut samples)
}

/// p50 of the SQL `COUNT(*)` query over `ITERS` timed runs (after one warmup).
fn measure_sql(st: &Supertable) -> Duration {
    let warm = st
        .reader()
        .query_sql("SELECT COUNT(*) FROM supertable")
        .expect("sql");
    black_box(warm);
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let batches = st
            .reader()
            .query_sql(black_box("SELECT COUNT(*) FROM supertable"))
            .expect("sql");
        samples.push(t0.elapsed());
        black_box(batches);
    }
    p50(&mut samples)
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

fn state_row(state: WorkloadState) -> Vec<Cell> {
    let (_dir, st) = build_supertable(state);
    let sampler = PeakSampler::start_default();
    let fts = measure_fts(&st).as_secs_f64() * NS_PER_SEC;
    let sql = measure_sql(&st).as_secs_f64() * NS_PER_SEC;
    let rss = sampler.stop_stats();
    let mut cells = vec![
        text(state.label()),
        metric(fts, fmt_time(fts), Better::Lower),
        metric(sql, fmt_time(sql), Better::Lower),
    ];
    cells.extend(rss_cells(rss));
    cells
}

pub fn run() {
    eprintln!(
        "[tombstone-overhead] {N_DOCS} docs / {APPEND_CHUNKS} superfiles; measuring FTS + SQL p50 across clean / one_percent / ten_percent_churned..."
    );
    let rows = vec![
        state_row(WorkloadState::Clean),
        state_row(WorkloadState::OnePercent),
        state_row(WorkloadState::TenPercentChurned),
    ];

    let mut report = Report::load("tombstone-overhead");
    report.emit(&Section {
        anchor: "bench/tombstone/overhead".into(),
        title: format!(
            "Tombstone overhead — reader-side filter warm path ({N_DOCS} docs, {APPEND_CHUNKS} superfiles)"
        ),
        note: "Per-query p50 across three tombstone states. `clean` is the empty-bitmap \
               short-circuit floor; `one_percent` tombstones the first 1% of each superfile's \
               docs (driven through the WAL `run_tombstone_phase` pipeline); `ten_percent_churned` \
               tombstones 10% then re-tombstones the same rows to exercise the CAS-loss / \
               bitmap-union path. FTS is a top-10 BM25 OR query for `alpha`; SQL is `COUNT(*)`. \
               Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "State".into(),
                "FTS p50".into(),
                "SQL COUNT(*) p50".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows,
        }],
    });
    report.save();
}
