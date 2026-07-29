// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Streaming recall-over-time diagnostic (concurrent ingest + cron optimize).
//!
//! Measures **recall@10 as the table grows**, rather than at a single
//! post-load instant. A fixed held-out query set is scored against an inline
//! running ground truth (per-query top-k heaps updated as each batch streams
//! by, so the corpus is never materialized), while a background cron fires
//! `optimize()` (drain + split) on a cadence. At every checkpoint ingest
//! pauses, the index is queried, and one table row is emitted:
//!
//! ```text
//! idx  prefix  recall@10  drained%  cells  over_cap
//! 1    100k    0.990      100%      8      0
//! ...
//! ```
//!
//! It **records** recall (no floor assertion) — dips are the datum. The
//! driver reuses the existing bench machinery wholesale: `tiers` for storage,
//! `ingest::supertable::options_for` for construction, the public
//! `vector_search` path via `executors::vector::SupertableVectorRead`, and the
//! `corpus` recall/ground-truth primitives. Only the streaming loop, the
//! running-heap ground truth, and this mode's wiring are new.
//!
//! Knobs (env vars):
//!   INFINO_BENCH_SUPERTABLE_DOCS          total docs to stream (default 10M)
//!   INFINO_BENCH_RECALL_CHECKPOINT_DOCS   measure every N docs (default 100_000)
//!   INFINO_BENCH_RECALL_OPTIMIZE_CADENCE  cron period in seconds (default 60)
//!   INFINO_BENCH_RECALL_QUERIES           held-out query count (default 100)
//!   INFINO_BENCH_RECALL_CELL_CAP          per-cell doc cap for `over_cap` (optional)
//!   plus the shared INFINO_BENCH_STORE / INFINO_BENCH_CELLS / INFINO_BENCH_WRITERS
//!   and `cell_split_doc_cap` via ./infino.yaml.
//!
//! Invoked as `cargo bench -- recall_while_ingest`.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arrow_array::{Array, Decimal128Array, RecordBatch};
use arrow_schema::Schema;
use infino::{OptimizeOptions, supertable::Supertable};
use rayon::prelude::*;

use crate::{
    corpus::{self, DIM, SequentialSyntheticCorpus},
    diag_common::{env_bool_default_true, env_u64, env_usize},
    executors::vector as exec_vec,
    ingest::supertable::{self as ingest, Modality, VEC_COLUMN},
    markdown::fmt_count,
    report::{Better, Block, Cell, Report, Section, metric, text},
    tiers,
};

const K: usize = 10;
const DEFAULT_CHECKPOINT_DOCS: usize = 100_000;
/// Upper bound on the per-append generation buffer, and thus the per-commit
/// doc count (each sub-batch appends then commits). The ingest sub-batch is
/// DERIVED as `min(this, remaining checkpoint)` — never a separate knob — so
/// the resident `flat` buffer stays ~constant (this × dim × 4B) no matter how
/// large `CHECKPOINT_DOCS` is. That keeps `CHECKPOINT_DOCS` a pure measure
/// interval and lets a single-shot run (`CHECKPOINT_DOCS` = total) work without
/// materializing the whole corpus in one buffer (OOM).
///
/// Fixed to the bulk bench's [`ingest::MAX_DOCS_PER_COMMIT`] so the first
/// commit — which bootstraps the immutable 256-cell global grid — trains on the
/// SAME sample size as the standard vector bench. A smaller first commit would
/// bootstrap the grid on less data and make streaming-vs-bulk recall a
/// different-grid comparison rather than an engine comparison.
const MAX_INGEST_BATCH_DOCS: usize = ingest::MAX_DOCS_PER_COMMIT;
const DEFAULT_OPTIMIZE_CADENCE_SECS: u64 = 60;
const DEFAULT_QUERIES: usize = 100;
/// Corpus seeds — matched to the ingest generators so the held-out queries
/// perturb real early corpus rows (well-defined, non-trivial ground truth).
const VEC_SEED: u64 = 1;
const TEXT_SEED: u64 = 1;
/// Held-out query perturbation seed + sigma (mirrors the vector bench).
const QUERY_SEED: u64 = 17;
const QUERY_SIGMA: f32 = 0.05;
/// Producer memory budget (steers the disk cache's post-commit madvise sweep).
const WRITER_MEMORY_BUDGET_BYTES: u64 = 8 << 30;

/// Coarse cell-probe widths swept by the breadth diagnostic (recall vs. how
/// many cells are probed). The table's current cell count is appended at the
/// call site so the sweep also probes every cell.
const BREADTH_SWEEP_NPROBE_STEPS: [usize; 5] = [2, 4, 8, 16, 32];
/// Cron poll granularity — checks the cadence deadline this often.
const CRON_POLL: Duration = Duration::from_millis(200);

// ─── Inline running ground truth ────────────────────────────────────────────

/// One scored candidate for a query's running top-k. Ordered by similarity
/// (higher dot = closer for L2-normalized cosine), ties broken by id.
#[derive(Clone, Copy)]
struct Cand {
    dot: f32,
    id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dot == other.dot && self.id == other.id
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dot.total_cmp(&other.dot).then(self.id.cmp(&other.id))
    }
}

/// Bounded per-query top-k, maintained as a min-heap so the weakest survivor
/// is evicted first. This is the running ground truth: because ingest is
/// append-only and id-ordered, the exact top-k over prefix `[0, N)` is the
/// merge of the previous heap with the exact top-k of each new batch.
struct HeldTopK {
    heap: BinaryHeap<Reverse<Cand>>,
}

impl HeldTopK {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::with_capacity(K + 1),
        }
    }

    fn offer(&mut self, dot: f32, id: u32) {
        let cand = Cand { dot, id };
        if self.heap.len() < K {
            self.heap.push(Reverse(cand));
        } else if let Some(Reverse(weakest)) = self.heap.peek()
            && cand > *weakest
        {
            self.heap.pop();
            self.heap.push(Reverse(cand));
        }
    }

    fn ids(&self) -> Vec<u32> {
        self.heap.iter().map(|Reverse(c)| c.id).collect()
    }
}

/// Fold a freshly generated batch's exact top-k into the running heaps, one
/// brute-force pass parallelized across queries (each query is independent, so
/// no re-read). `base` is the dense id of the batch's first row.
fn update_heaps(heaps: &mut [HeldTopK], queries: &[Vec<f32>], flat: &[f32], base: u32, len: usize) {
    heaps
        .par_iter_mut()
        .zip(queries.par_iter())
        .for_each(|(heap, q)| {
            for j in 0..len {
                let v = &flat[j * DIM..(j + 1) * DIM];
                let mut dot = 0f32;
                for d in 0..DIM {
                    dot += v[d] * q[d];
                }
                heap.offer(dot, base + j as u32);
            }
        });
}

// ─── Held-out queries + batch construction ──────────────────────────────────

/// Build `n_queries` held-out query vectors by perturbing the first
/// `n_queries` corpus rows (streamed transiently, then discarded). Reuses the
/// vector bench's realistic-query generator so recall is meaningful at the
/// engine's default routing.
fn build_queries(n_cent: usize, n_queries: usize) -> Vec<Vec<f32>> {
    let mut src = SequentialSyntheticCorpus::new(n_cent, VEC_SEED, TEXT_SEED, true);
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    src.fill_chunk_modality(n_queries, &mut titles, &mut flat, false, true);
    corpus::generate_realistic_queries(&flat, n_queries, n_queries, QUERY_SEED, true, QUERY_SIGMA)
}

/// One append batch straight off the streamed `flat` (no corpus retained),
/// reusing the ingest path's `vector_array` builder so the column layout is
/// byte-identical to what `options_for(Modality::Vector, _)` expects.
fn vector_batch(schema: &Arc<Schema>, flat: &[f32], len: usize) -> RecordBatch {
    RecordBatch::try_new(
        schema.clone(),
        vec![ingest::vector_array(&flat[..len * DIM])],
    )
    .expect("vector RecordBatch")
}

// ─── Diagnostic columns (best-effort, from the hidden-index manifest) ─────────

struct HiddenStats {
    cells: Option<usize>,
    drained_pct: Option<f64>,
    over_cap: Option<usize>,
}

/// Read `cells`, `drained%`, and (when a cap is supplied) `over_cap` from the
/// hidden vector-index manifest. Mirrors `log_hidden_stats` /
/// `current_routing_phase` in the vector bench; returns `None` fields when the
/// hidden index has not been created/drained yet.
fn hidden_stats(consumer: &Supertable, cell_cap: Option<u64>) -> HiddenStats {
    let Some(hidden) = consumer.vector_index_table() else {
        return HiddenStats {
            cells: None,
            drained_pct: None,
            over_cap: None,
        };
    };
    // Per-cell row counts from the hidden manifest (the same walk
    // `log_hidden_stats` does); recall only needs the per-cell totals.
    let mut rows_by_cell: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for entry in hidden.pinned_reader().manifest().get_all_superfiles() {
        for summary in entry.vector_summary.values() {
            for cell in &summary.cells {
                if let Some(cell_id) = cell.cell_id {
                    *rows_by_cell.entry(cell_id).or_default() += cell
                        .clusters
                        .counts
                        .iter()
                        .map(|c| u64::from(*c))
                        .sum::<u64>();
                }
            }
        }
    }
    let cells = rows_by_cell.len();
    let over_cap = cell_cap.map(|cap| rows_by_cell.values().filter(|&&rows| rows > cap).count());

    // drained% = user superfiles whose birth version is in the drained set.
    let user_reader = consumer.reader().expect("reader");
    let user_sfs = user_reader.manifest().get_all_superfiles();
    let drained_ranges = hidden.pinned_reader().manifest().get_drained_ranges();
    let drained_pct = if user_sfs.is_empty() {
        Some(100.0)
    } else {
        let drained = user_sfs
            .iter()
            .filter(|e| drained_ranges.contains(e.birth_version))
            .count();
        Some(100.0 * drained as f64 / user_sfs.len() as f64)
    };

    HiddenStats {
        cells: Some(cells),
        drained_pct,
        over_cap,
    }
}

// ─── Stable `_id` → dense map (rebuilt per checkpoint) ───────────────────────

/// The engine mints a 128-bit Snowflake `_id` per row (NOT the dense ingest
/// position), so the query hits — which speak `_id` — must be translated to
/// the dense ids the running heaps hold. Because Snowflake ids are minted by
/// multiple parallel writers, `_id` order is NOT strictly ingest order, so the
/// map is rebuilt from a full `_id` scan each checkpoint rather than extended by
/// an `_id > last_max` prune (which would skip rows whose id sorts below a prior
/// batch's max and later panic `measure` with an unmapped `_id`). Dense ids are
/// assigned in `_id ASC` order, matching the heaps.
struct IdMap {
    /// `_id` → dense ingest position.
    to_dense: std::collections::HashMap<i128, u32>,
    /// Next dense id to assign — equals the count ingested so far.
    next_dense: u32,
}

impl IdMap {
    fn new() -> Self {
        Self {
            to_dense: std::collections::HashMap::new(),
            next_dense: 0,
        }
    }

    /// Pull the `_id`s appended since the last call and assign them dense ids
    /// in `_id` (ingest) order. First call reads the whole (tiny) table; later
    /// calls prune to `_id > last_max`.
    fn extend(&mut self, consumer: &Supertable) {
        // Full rebuild each checkpoint. The engine mints 128-bit Snowflake `_id`s
        // across parallel writers, so `_id` order is NOT strictly ingest order — an
        // incremental `_id > last_max` prune silently skips rows whose id sorts
        // below a prior batch's max, and a later `vector_search` hit on a skipped
        // row then panics `measure` with an unmapped `_id`. A full scan of the one
        // `_id` column is O(n) but correct at any scale.
        self.to_dense.clear();
        self.next_dense = 0;
        let batches = consumer
            .reader()
            .expect("reader")
            .query_sql("SELECT _id FROM supertable ORDER BY _id")
            .expect("SELECT _id for id map");
        for batch in &batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id column is Decimal128");
            for i in 0..col.len() {
                self.to_dense.insert(col.value(i), self.next_dense);
                self.next_dense += 1;
            }
        }
    }
}

// ─── Recall at a checkpoint ──────────────────────────────────────────────────

/// Mean recall@k over the held-out queries at the current prefix: query the
/// public `vector_search` path (engine-default routing), translate the returned
/// `_id`s to dense via the incremental [`IdMap`], and intersect with the
/// running heaps. Reuses `id_scores_from_vector_search` + `recall_at_k`; the
/// map is borrowed (never cloned) so there is no per-checkpoint O(N) cost.
fn measure_recall(
    consumer: &Supertable,
    queries: &[Vec<f32>],
    heaps: &[HeldTopK],
    id_map: &IdMap,
    nprobe: usize,
) -> f32 {
    let reader = consumer.reader().expect("reader");
    // nprobe == ENGINE_DEFAULT (0) keeps engine-default routing; a positive
    // value overrides the coarse cell-probe width (breadth) via search_opts.
    // INFINO_BENCH_RERANK_MULT (diagnostic) overrides the Sq8 rerank shortlist
    // depth (candidates ≈ rerank_mult × k); default ENGINE_DEFAULT → 256. Tests
    // whether a deeper within-cell rerank recovers recall at COARSE grid cells.
    let rerank_mult = std::env::var("INFINO_BENCH_RERANK_MULT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(exec_vec::ENGINE_DEFAULT);
    let opts = exec_vec::search_opts(nprobe, rerank_mult);
    // Misrank audit (diagnostic): retrieve INFINO_BENCH_FETCH_K results but still
    // score against the GT top-K. fetch_k=1000 vs the default K=10 answers: are
    // the missed true neighbors MISRANKED (present in the top-1000 → scoring
    // precision) or NOT RETRIEVED (absent from 1000 → coverage/routing)?
    let fetch_k = std::env::var("INFINO_BENCH_FETCH_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(K);
    let mut sum = 0f32;
    for (q, heap) in queries.iter().zip(heaps) {
        let batches = reader
            .vector_search(VEC_COLUMN, q, fetch_k, opts, None, None)
            .expect("recall vector_search");
        let hits: Vec<(u32, f32)> = corpus::id_scores_from_vector_search(&batches)
            .into_iter()
            .map(|(id, score)| {
                let dense = *id_map
                    .to_dense
                    .get(&id)
                    .unwrap_or_else(|| panic!("vector_search returned unmapped _id {id}"));
                (dense, score)
            })
            .collect();
        sum += corpus::recall_at_k(&hits, &heap.ids());
    }
    sum / queries.len() as f32
}

/// Miss-trace: for each held-out query, self-query every MISSED GT doc with its
/// own retained vector. `self_found` (doc finds itself) ⇒ it IS reachable, so
/// the original miss is routing/boundary (the doc sits in a cell the query
/// doesn't probe); `self_missing` (doc can't even find itself) ⇒ a real index
/// defect. Resolves "how can it be missing if we probed everything".
fn trace_misses(
    consumer: &Supertable,
    queries: &[Vec<f32>],
    heaps: &[HeldTopK],
    id_map: &IdMap,
    retained: &[f32],
) {
    let reader = consumer.reader().expect("reader");
    let opts = exec_vec::search_opts(exec_vec::ENGINE_DEFAULT, exec_vec::ENGINE_DEFAULT);
    let n_retained = retained.len() / DIM;

    // Measurement soundness: a PURE bench-side exact brute-force (dot over every
    // retained vector, no engine/codec/IVF) vs the GT heap. It MUST be ~1.0 — if
    // not, the GT pipeline itself is inconsistent and every engine recall number
    // was measured against a broken baseline. Parallel over queries.
    let bf: f32 = queries
        .par_iter()
        .zip(heaps.par_iter())
        .map(|(q, heap)| {
            let mut top: Vec<(f32, u32)> = (0..n_retained)
                .map(|d| {
                    let v = &retained[d * DIM..(d + 1) * DIM];
                    let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                    (dot, d as u32)
                })
                .collect();
            top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let bf_ids: std::collections::HashSet<u32> =
                top.iter().take(K).map(|(_, id)| *id).collect();
            let gt = heap.ids();
            let hit = gt.iter().filter(|id| bf_ids.contains(id)).count();
            hit as f32 / gt.len().max(1) as f32
        })
        .sum::<f32>()
        / queries.len() as f32;
    eprintln!(
        "[brute-force-check] pure exact recall vs GT = {bf:.4} (MUST be ~1.0 if GT is sound)"
    );

    let (mut total_miss, mut self_found, mut self_missing, mut samples) =
        (0usize, 0usize, 0usize, 0usize);
    for (qi, (q, heap)) in queries.iter().zip(heaps).enumerate() {
        let batches = reader
            .vector_search(VEC_COLUMN, q, K, opts, None, None)
            .expect("miss-trace query");
        let returned: std::collections::HashSet<u32> =
            corpus::id_scores_from_vector_search(&batches)
                .into_iter()
                .filter_map(|(id, _)| id_map.to_dense.get(&id).copied())
                .collect();
        for gt_id in heap.ids() {
            if returned.contains(&gt_id) || (gt_id as usize) >= n_retained {
                continue;
            }
            total_miss += 1;
            let dv = &retained[gt_id as usize * DIM..(gt_id as usize + 1) * DIM];
            let sb = reader
                .vector_search(VEC_COLUMN, dv, K, opts, None, None)
                .expect("self-query");
            let self_hits: Vec<u32> = corpus::id_scores_from_vector_search(&sb)
                .into_iter()
                .filter_map(|(id, _)| id_map.to_dense.get(&id).copied())
                .collect();
            let found = self_hits.contains(&gt_id);
            if found {
                self_found += 1;
            } else {
                self_missing += 1;
            }
            if samples < 6 {
                eprintln!(
                    "[miss-trace] q{qi} missed doc {gt_id}: self-query found_self={found} rank1={:?}",
                    self_hits.first()
                );
                samples += 1;
            }
        }
    }
    eprintln!(
        "[miss-trace] total_miss={total_miss} self_found(reachable→routing)={self_found} self_missing(unreachable→defect)={self_missing}"
    );
}

fn fmt_opt_pct(v: Option<f64>) -> String {
    v.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "?".into())
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "?".into())
}

/// Report cell for an optional percentage (Δ-tracked when present).
fn pct_cell(v: Option<f64>) -> Cell {
    match v {
        Some(p) => metric(p, format!("{p:.0}%"), Better::Higher),
        None => text("?"),
    }
}

/// Report cell for an optional count (Δ-tracked when present).
fn count_cell(v: Option<usize>, better: Better) -> Cell {
    match v {
        Some(n) => metric(n as f64, n.to_string(), better),
        None => text("?"),
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

pub fn run() {
    // Same backing-store contract as the supertable bench (RustFS default, or
    // S3/Azure/GCS/local via INFINO_BENCH_STORE).
    if let Err(reason) = tiers::supertable_backend_check() {
        eprintln!("[recall_while_ingest] skipped: {reason}");
        return;
    }

    let total_docs = ingest::n_docs();
    // `.max(1)` guards the loop-critical knobs: a 0 checkpoint would spin the
    // outer loop forever, and 0 queries would divide by zero in the recall mean.
    let checkpoint = env_usize(
        "INFINO_BENCH_RECALL_CHECKPOINT_DOCS",
        DEFAULT_CHECKPOINT_DOCS,
    )
    .max(1);
    let cadence = Duration::from_secs(env_u64(
        "INFINO_BENCH_RECALL_OPTIMIZE_CADENCE",
        DEFAULT_OPTIMIZE_CADENCE_SECS,
    ));
    let n_queries = env_usize("INFINO_BENCH_RECALL_QUERIES", DEFAULT_QUERIES).max(1);
    let force_sync = env_bool_default_true("INFINO_BENCH_RECALL_FORCE_OPTIMIZE_AFTER_BATCH");
    let debug = std::env::var_os("INFINO_BENCH_RECALL_DEBUG").is_some();
    // The `over_cap` column tracks the SAME cap the engine splits on
    // (`vector.cell_split_doc_cap`, from ./infino.yaml), so an over-cap row
    // means the engine's split path *should* have fired. An explicit
    // INFINO_BENCH_RECALL_CELL_CAP overrides only the reported column.
    let engine_cap = infino::config::global().vector.cell_split_doc_cap;
    let report_cap = std::env::var("INFINO_BENCH_RECALL_CELL_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&c| c > 0)
        .or(Some(engine_cap));
    let n_cent = corpus::n_cent(total_docs);

    let optimize_desc = if force_sync {
        "synchronous optimize() after each batch".to_string()
    } else {
        format!("wall-clock cron optimize() every {}s", cadence.as_secs())
    };
    eprintln!(
        "[recall_while_ingest] streaming {} docs in {}-doc checkpoints, {optimize_desc}, \
         {} held-out queries, engine cell_split_doc_cap={engine_cap}",
        fmt_count(total_docs),
        fmt_count(checkpoint),
        n_queries,
    );
    if debug {
        eprintln!(
            "[recall_while_ingest] DEBUG cwd={:?}  INFINO_BENCH_CELLS={:?}",
            std::env::current_dir().ok(),
            std::env::var("INFINO_BENCH_CELLS").ok(),
        );
    }

    // Held-out queries + running ground-truth heaps.
    let queries = build_queries(n_cent, n_queries);
    let mut heaps: Vec<HeldTopK> = (0..n_queries).map(|_| HeldTopK::new()).collect();

    // Backing store (reuse the supertable fixture so INFINO_BENCH_STORE applies).
    let fixture = tiers::block_on(tiers::supertable_storage_fixture());
    let storage = Arc::clone(&fixture.storage);
    eprintln!(
        "[recall_while_ingest] backing store: {}",
        fixture.storage_label
    );

    // Shared options builder (schema, cell counts, INFINO_BENCH_CELLS, pools all
    // handled there), plus one ingest disk cache reused by every handle.
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage));
    let build_opts = || {
        ingest::options_for(Modality::Vector, Some(Arc::clone(&storage)))
            .with_disk_cache(cache.clone())
            .with_memory_budget(WRITER_MEMORY_BUDGET_BYTES)
            .with_cache_prepopulation(false)
    };
    let mut st = Supertable::create(build_opts()).expect("create supertable");
    let schema = ingest::schema_for(Modality::Vector);

    // Cron handles: created in both modes but consumed only by the wall-clock
    // cron thread, which is spawned lazily after the first reopen (below).
    let stop = Arc::new(AtomicBool::new(false));
    let busy = Arc::new(AtomicBool::new(false));
    let ingested = Arc::new(AtomicUsize::new(0));
    let mut cron: Option<thread::JoinHandle<()>> = None;

    // Report table accumulated across checkpoints, emitted at the end.
    let mut report = Report::load("recall_while_ingest");
    let mut rows: Vec<Vec<Cell>> = Vec::new();

    // Streaming ingest + measure loop.
    let mut stream = SequentialSyntheticCorpus::new(n_cent, VEC_SEED, TEXT_SEED, true);
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    let mut id_map = IdMap::new();
    // Miss-trace (diagnostic, small scale only): retain every ingested vector so
    // that at a checkpoint we can SELF-QUERY each missed GT doc with its own
    // vector — if a doc can't find ITSELF, it's a real index defect; if it can,
    // the miss is routing/boundary (the doc sits in a cell the query doesn't
    // probe), not a measurement artifact.
    let miss_trace = std::env::var_os("INFINO_BENCH_MISS_TRACE").is_some();
    let mut retained: Vec<f32> = Vec::new();
    let mut n = 0usize;
    let mut idx = 0usize;
    let mut docs_at_last_opt = 0usize;
    let mut reopened = false;

    eprintln!("[recall_while_ingest] idx  prefix  recall@10  drained%  cells  over_cap");
    while n < total_docs {
        let checkpoint_len = checkpoint.min(total_docs - n);
        // Ingest the checkpoint in bounded sub-batches so the generation buffer
        // stays ~constant (MAX_INGEST_BATCH_DOCS), independent of CHECKPOINT_DOCS.
        // The sub-batch is DERIVED (min with the remaining checkpoint), so it
        // always evenly divides the checkpoint and the measure lands exactly on
        // the boundary. Per sub-batch: fill → score GT → append → discard. A
        // fresh writer per sub-batch avoids a stale manifest view after the
        // previous iteration's optimize().
        let mut off = 0usize;
        while off < checkpoint_len {
            let sub = MAX_INGEST_BATCH_DOCS.min(checkpoint_len - off);
            stream.fill_chunk_modality(sub, &mut titles, &mut flat, false, true);
            if miss_trace {
                retained.extend_from_slice(&flat[..sub * DIM]);
            }
            update_heaps(&mut heaps, &queries, &flat, (n + off) as u32, sub);
            let batch = vector_batch(&schema, &flat, sub);
            {
                let mut writer = st.writer().expect("writer");
                writer.append(&batch).expect("append");
                writer.commit().expect("commit");
            }
            off += sub;
            ingested.store(n + off, Ordering::Relaxed);
        }
        n += checkpoint_len;
        idx += 1;

        // Re-open once the first batch is committed. The create-time handle
        // built its hidden vector index from an EMPTY user manifest, so the
        // hidden options carry no `VectorCell` partition strategy — and
        // `optimize()` therefore never enters the over-cap cell-split path
        // (`is_hidden_vector_index_table` checks the options). Re-opening from
        // storage, with the first batch committed, trains the hidden cell grid
        // and stamps `VectorCell` into the hidden options, so subsequent
        // `optimize()`s split over-cap cells. The reopened handle serves the
        // rest of the run (append + optimize + query on one handle).
        if !reopened {
            st = Supertable::open(build_opts()).expect("reopen supertable");
            reopened = true;
            if !force_sync {
                let st = st.clone();
                let stop = Arc::clone(&stop);
                let busy = Arc::clone(&busy);
                let ingested = Arc::clone(&ingested);
                cron = Some(
                    thread::Builder::new()
                        .name("recall-optimize-cron".into())
                        .spawn(move || {
                            let mut last_fire = Instant::now();
                            let mut docs_at_last = 0usize;
                            while !stop.load(Ordering::Relaxed) {
                                thread::sleep(CRON_POLL);
                                if last_fire.elapsed() < cadence {
                                    continue;
                                }
                                // Re-entrancy guard: skip if a previous optimize
                                // is still running (single cron thread can't
                                // stack, but the guard makes the contract explicit).
                                if busy
                                    .compare_exchange(
                                        false,
                                        true,
                                        Ordering::AcqRel,
                                        Ordering::Acquire,
                                    )
                                    .is_err()
                                {
                                    continue;
                                }
                                last_fire = Instant::now();
                                let docs_now = ingested.load(Ordering::Relaxed);
                                let t0 = Instant::now();
                                match st.optimize(&OptimizeOptions::default()) {
                                    Ok(_) => eprintln!(
                                        "[recall_while_ingest] cron optimize() done in {:.1}s ({} docs since last, {} total)",
                                        t0.elapsed().as_secs_f64(),
                                        fmt_count(docs_now.saturating_sub(docs_at_last)),
                                        fmt_count(docs_now),
                                    ),
                                    Err(e) => {
                                        eprintln!("[recall_while_ingest] cron optimize() failed: {e}")
                                    }
                                }
                                docs_at_last = docs_now;
                                busy.store(false, Ordering::Release);
                            }
                        })
                        .expect("spawn recall-optimize-cron"),
                );
            }
        }

        // Synchronous optimize (default): small per-batch drain + split right
        // after ingest, so each row reflects the at-rest, healed state.
        if force_sync {
            let t0 = Instant::now();
            match st.optimize(&OptimizeOptions::default()) {
                Ok(_) => eprintln!(
                    "[recall_while_ingest] sync optimize() done in {:.1}s ({} docs since last, {} total)",
                    t0.elapsed().as_secs_f64(),
                    fmt_count(n.saturating_sub(docs_at_last_opt)),
                    fmt_count(n),
                ),
                Err(e) => eprintln!("[recall_while_ingest] sync optimize() failed: {e}"),
            }
            docs_at_last_opt = n;
        }

        // Extend the stable-id → dense map with just this batch's new rows
        // (pruned query, never a whole-table rescan).
        id_map.extend(&st);

        // Ingest is paused here (single loop thread) so the prefix is crisp.
        if n < K {
            continue;
        }
        let recall = measure_recall(&st, &queries, &heaps, &id_map, exec_vec::ENGINE_DEFAULT);
        let stats = hidden_stats(&st, report_cap);
        eprintln!(
            "[recall_while_ingest] {idx:<4} {:<7} {recall:<9.3} {:<9} {:<6} {}",
            fmt_count(n),
            fmt_opt_pct(stats.drained_pct),
            fmt_opt_usize(stats.cells),
            fmt_opt_usize(stats.over_cap),
        );
        // Breadth diagnostic: sweep the coarse cell-probe width against this
        // built table. If recall climbs toward 1.0 as more cells are probed,
        // the loss is coverage/route-fidelity (breadth), not within-cell depth.
        let cells_now = stats.cells.unwrap_or(0);
        for np in BREADTH_SWEEP_NPROBE_STEPS
            .into_iter()
            .chain([cells_now])
            .filter(|&x| x > 0)
        {
            let r = measure_recall(&st, &queries, &heaps, &id_map, np);
            eprintln!("[breadth-sweep] nprobe={np:<5} recall@10={r:.3}");
        }
        if miss_trace {
            trace_misses(&st, &queries, &heaps, &id_map, &retained);
        }
        rows.push(vec![
            text(fmt_count(n)),
            metric(recall as f64, format!("{recall:.3}"), Better::Higher),
            pct_cell(stats.drained_pct),
            count_cell(stats.cells, Better::Higher),
            count_cell(stats.over_cap, Better::Lower),
        ]);
    }

    // Tear down: stop the cron (if any), then clean up a remote prefix.
    stop.store(true, Ordering::Relaxed);
    if let Some(cron) = cron {
        cron.join().expect("join recall-optimize-cron");
    }

    if !rows.is_empty() {
        report.emit(&Section {
            anchor: "bench/recall_while_ingest/over-time".into(),
            title: format!(
                "Recall over time — streaming ingest + {} ({} docs, {}-doc checkpoints, {} queries)",
                if force_sync {
                    "synchronous optimize".to_string()
                } else {
                    format!("cron optimize every {}s", cadence.as_secs())
                },
                fmt_count(total_docs),
                fmt_count(checkpoint),
                n_queries,
            ),
            note: format!(
                "recall@10 vs an inline running brute-force ground truth (per-query top-k heaps \
                 updated as each batch streams by — the corpus is never materialized), measured \
                 after each checkpoint's ingest + optimize. `drained%` = user superfiles drained \
                 into the hidden cell index; `cells` = live hidden cells; `over_cap` = cells above \
                 the split cap ({engine_cap}). A measurement (dips are the datum), not a gate. \
                 Δ is vs the previous run."
            ),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "prefix".into(),
                    "recall@10".into(),
                    "drained%".into(),
                    "cells".into(),
                    "over_cap".into(),
                ],
                rows,
            }],
        });
        report.save();
    }

    drop(st);
    drop(cache);
    drop(cache_dir);
    if let Some(cleanup) = &fixture.cleanup {
        eprintln!("[recall_while_ingest] cleaning up object-store prefix...");
        tiers::cleanup_prefix(cleanup);
    }
}
