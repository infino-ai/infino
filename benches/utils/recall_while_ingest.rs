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
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arrow_array::{Array, Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::Schema;
use infino::{OptimizeOptions, supertable::Supertable};
use rayon::prelude::*;

use crate::{
    corpus::{
        self, DIM, SequentialSyntheticCorpus,
        grading::{read_f32, read_u32, read_u64},
    },
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

    /// `(id, score)` pairs currently held — for GT-bin serialization.
    fn entries(&self) -> Vec<(u32, f32)> {
        self.heap.iter().map(|Reverse(c)| (c.id, c.dot)).collect()
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

// ─── Ground-truth bin (persist / reload across resumes) ──────────────────────

/// On-disk GT-bin format version. Bumped on any header/layout change so an
/// older bin is refused rather than misparsed.
const GT_BIN_VERSION: u32 = 1;
/// Fixed header size in bytes: magic(4) + version(4) + m(8) + n_queries(8) +
/// n_cent(8) + dim(4) + k(4) + vec_seed(8) + query_seed(8).
const GT_BIN_HEADER_BYTES: usize = 56;

/// A loaded GT bin: the doc-count it covers, the ORIGINAL build's cluster count
/// (provenance — the header is authoritative), and the rebuilt per-query heaps.
struct GtBin {
    m: usize,
    n_cent: usize,
    heaps: Vec<HeldTopK>,
}

/// Why a GT bin could not be loaded. The two cases must be handled
/// differently on resume: `Corrupt` (io error / bad magic / truncation) is
/// recoverable — rebuild by replay; `Incompatible` means the bin was written
/// by a build with different query-DETERMINING constants (format version /
/// dim / k / seeds), so its heaps are keyed to different queries. That is
/// provenance the resume cannot override, so it must abort rather than score
/// against the wrong ground truth.
#[derive(Debug)]
enum GtBinError {
    Corrupt(String),
    Incompatible(String),
}

/// Serialize the running GT heaps to `path` atomically (temp + rename) so a
/// crash never leaves a half-written bin. The header carries the provenance a
/// reload needs to reject an incompatible bin.
fn gt_bin_write(
    path: &str,
    heaps: &[HeldTopK],
    m: usize,
    n_queries: usize,
    n_cent: usize,
) -> io::Result<()> {
    let mut b: Vec<u8> = Vec::with_capacity(GT_BIN_HEADER_BYTES + heaps.len() * (K * 8 + 4));
    b.extend_from_slice(b"GTB1");
    b.extend_from_slice(&GT_BIN_VERSION.to_le_bytes());
    b.extend_from_slice(&(m as u64).to_le_bytes());
    b.extend_from_slice(&(n_queries as u64).to_le_bytes());
    b.extend_from_slice(&(n_cent as u64).to_le_bytes());
    b.extend_from_slice(&(DIM as u32).to_le_bytes());
    b.extend_from_slice(&(K as u32).to_le_bytes());
    b.extend_from_slice(&VEC_SEED.to_le_bytes());
    b.extend_from_slice(&QUERY_SEED.to_le_bytes());
    for h in heaps {
        let e = h.entries();
        b.extend_from_slice(&(e.len() as u32).to_le_bytes());
        for (id, score) in e {
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&score.to_le_bytes());
        }
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, &b)?;
    fs::rename(&tmp, path)
}

/// Header + heaps parsed from a bin's bytes (past the magic). Truncation
/// surfaces as the reader's `io::Error`; compatibility is judged by the caller.
struct ParsedGtBin {
    version: u32,
    m: usize,
    n_queries: usize,
    n_cent: usize,
    dim: usize,
    k: usize,
    vec_seed: u64,
    query_seed: u64,
    heaps: Vec<HeldTopK>,
}

/// Parse `bytes` (everything after the 4-byte magic) via the shared `grading`
/// cursor readers. Any short read is a truncation `io::Error`.
fn parse_gt_bin(bytes: &[u8]) -> io::Result<ParsedGtBin> {
    let mut cur: &[u8] = bytes;
    let version = read_u32(&mut cur)?;
    let m = read_u64(&mut cur)? as usize;
    let n_queries = read_u64(&mut cur)? as usize;
    let n_cent = read_u64(&mut cur)? as usize;
    let dim = read_u32(&mut cur)? as usize;
    let k = read_u32(&mut cur)? as usize;
    let vec_seed = read_u64(&mut cur)?;
    let query_seed = read_u64(&mut cur)?;
    let mut heaps = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let len = read_u32(&mut cur)? as usize;
        let mut h = HeldTopK::new();
        for _ in 0..len {
            let id = read_u32(&mut cur)?;
            let score = read_f32(&mut cur)?;
            h.offer(score, id);
        }
        heaps.push(h);
    }
    Ok(ParsedGtBin {
        version,
        m,
        n_queries,
        n_cent,
        dim,
        k,
        vec_seed,
        query_seed,
        heaps,
    })
}

/// Load a GT bin. `n_cent` is read from the header as PROVENANCE — the original
/// build's cluster count, which a resume cannot re-derive from `COUNT(*)` (a run
/// that crashed mid-target sits in a different doc-count band than its target).
/// The query-DETERMINING constants (format version, `n_queries`, dim, k, seeds)
/// ARE validated against this run: a mismatch is [`GtBinError::Incompatible`] —
/// the heaps are keyed to different queries, so the caller must abort rather
/// than score against the wrong ground truth. Bad magic / truncation / io are
/// [`GtBinError::Corrupt`], safely recoverable by a replay rebuild.
fn gt_bin_read(path: &str, n_queries: usize) -> Result<GtBin, GtBinError> {
    let b = fs::read(path).map_err(|e| GtBinError::Corrupt(format!("read {path}: {e}")))?;
    if b.get(0..4) != Some(b"GTB1".as_slice()) {
        return Err(GtBinError::Corrupt(format!("{path}: bad magic")));
    }
    let p = parse_gt_bin(&b[4..])
        .map_err(|e| GtBinError::Corrupt(format!("{path}: truncated ({e})")))?;
    if p.version != GT_BIN_VERSION {
        return Err(GtBinError::Incompatible(format!(
            "{path}: format version {} != {GT_BIN_VERSION}",
            p.version
        )));
    }
    if p.n_queries != n_queries
        || p.dim != DIM
        || p.k != K
        || p.vec_seed != VEC_SEED
        || p.query_seed != QUERY_SEED
    {
        return Err(GtBinError::Incompatible(format!(
            "{path}: built with n_queries={} dim={} k={} vseed={} qseed={}; \
             this run uses n_queries={n_queries} dim={DIM} k={K} vseed={VEC_SEED} qseed={QUERY_SEED}",
            p.n_queries, p.dim, p.k, p.vec_seed, p.query_seed
        )));
    }
    Ok(GtBin {
        m: p.m,
        n_cent: p.n_cent,
        heaps: p.heaps,
    })
}

/// Per-checkpoint bin path: `{base}.M{count}.bin`, so the covered doc-count is
/// visible in the filename (`ls`) without reading the header. Immutable per
/// checkpoint (no overwrite) — a history of oracles like numbered manifests.
fn gt_bin_ckpt_path(base: &str, count: usize) -> String {
    format!("{base}.M{count}.bin")
}

/// Invoke `f(count, path)` for every persisted `{base}.M{count}.bin`. The
/// dir-scan + filename parse lives here so latest-selection and prune share one
/// definition — two copies of the parse is exactly the drift that bites later.
fn for_each_bin(base: &str, mut f: impl FnMut(usize, PathBuf)) {
    let base_path = Path::new(base);
    let dir = base_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let Some(stem) = base_path.file_name().and_then(|s| s.to_str()) else {
        return;
    };
    let prefix = format!("{stem}.M");
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".bin") else {
            continue;
        };
        let Ok(count) = digits.parse::<usize>() else {
            continue;
        };
        f(count, entry.path());
    }
}

/// Newest persisted bin for `base` — the `{base}.M{count}.bin` with the largest
/// `count` — as `(count, full_path)`. `None` if none exist yet.
fn gt_bin_latest(base: &str) -> Option<(usize, String)> {
    let mut best: Option<(usize, String)> = None;
    for_each_bin(base, |count, path| {
        if best.as_ref().is_none_or(|(bc, _)| count > *bc) {
            best = Some((count, path.to_string_lossy().into_owned()));
        }
    });
    best
}

/// Delete every `{base}.M{count}.bin` except `keep` — the newest bin fully
/// supersedes older ones (resume always loads the highest count). Call this
/// only AFTER the `keep` bin is durably written, so a crash never leaves zero
/// valid bins. Remove errors are ignored (a stray extra bin is harmless).
fn gt_bin_prune(base: &str, keep: usize) {
    for_each_bin(base, |count, path| {
        if count != keep {
            let _ = fs::remove_file(path);
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
/// byte-identical to what `options_for(Modality::Vector, _)` expects. The
/// `Modality::Vector` schema carries the filter-bucket column ahead of the
/// vector, so the batch mirrors that field order — bucket terms are derived
/// from each row's global doc id (`doc_base + i`), exactly as the bulk ingest
/// path stamps them.
fn vector_batch(schema: &Arc<Schema>, flat: &[f32], len: usize, doc_base: usize) -> RecordBatch {
    let buckets: Vec<String> = (doc_base..doc_base + len)
        .map(ingest::vector_filter_bucket_term)
        .collect();
    let bucket_col = Arc::new(LargeStringArray::from(
        buckets.iter().map(String::as_str).collect::<Vec<_>>(),
    )) as Arc<dyn Array>;
    RecordBatch::try_new(
        schema.clone(),
        vec![bucket_col, ingest::vector_array(&flat[..len * DIM])],
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
    // INFINO_BENCH_RERANK_MULT (diagnostic) overrides the Sq8 rerank shortlist
    // depth (candidates ≈ rerank_mult × k); default ENGINE_DEFAULT → 256. Tests
    // whether a deeper within-cell rerank recovers recall at COARSE grid cells.
    let rerank_mult = std::env::var("INFINO_BENCH_RERANK_MULT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(exec_vec::ENGINE_DEFAULT);
    measure_recall_rm(consumer, queries, heaps, id_map, nprobe, rerank_mult)
}

/// [`measure_recall`] with an explicit rerank multiplier — the oracle rung
/// forces `ceil(rows / K)` so the budget covers every row regardless of the
/// env override.
fn measure_recall_rm(
    consumer: &Supertable,
    queries: &[Vec<f32>],
    heaps: &[HeldTopK],
    id_map: &IdMap,
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let reader = consumer.reader().expect("reader");
    // nprobe == ENGINE_DEFAULT (0) keeps engine-default routing; a positive
    // value overrides the coarse cell-probe width (breadth) via search_opts.
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
    // Optional on-disk ground-truth bin, given as a BASE prefix. Each
    // checkpoint writes `{base}.M{count}.bin` (count = docs covered, visible in
    // the filename); resume loads the highest-count one. Lets a grown/resumed
    // run skip the [0, M) GT replay, and makes a crash mid-run cheap to recover.
    let gt_bin_path = std::env::var("INFINO_BENCH_GT_BIN")
        .ok()
        .filter(|s| !s.is_empty());
    // Resume against an existing table: gated on INFINO_BENCH_RESUME_INGEST.
    // RESUME_INGEST set without a non-empty INFINO_BENCH_EXISTING_PREFIX is a
    // hard error, NOT a silent fall-through to CREATE — the caller declared
    // intent to resume, and CREATE would mint a fresh bucket and full-rebuild.
    let resume = if std::env::var_os("INFINO_BENCH_RESUME_INGEST").is_some() {
        let prefix = std::env::var("INFINO_BENCH_EXISTING_PREFIX").unwrap_or_default();
        assert!(
            !prefix.is_empty(),
            "INFINO_BENCH_RESUME_INGEST is set but INFINO_BENCH_EXISTING_PREFIX is missing/empty; \
             refusing to fall through to a fresh CREATE"
        );
        true
    } else {
        false
    };

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

    // Backing store (reuse the supertable fixture so INFINO_BENCH_STORE applies).
    // On resume, open the existing prefix instead of minting a fresh one.
    let fixture = if resume {
        tiers::block_on(tiers::existing_supertable_storage_fixture())
            .expect("INFINO_BENCH_RESUME_INGEST requires a non-empty INFINO_BENCH_EXISTING_PREFIX")
    } else {
        tiers::block_on(tiers::supertable_storage_fixture())
    };
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
    let mut st = if resume {
        Supertable::open(build_opts()).expect("open existing supertable for resume")
    } else {
        Supertable::create(build_opts()).expect("create supertable")
    };
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

    // On resume, discover the ORIGINAL build size from the committed row count
    // (`m`) — we do NOT ask the caller to pass it. The corpus generator and
    // held-out queries are keyed to the original build's cluster count, so the
    // grid/query cluster count tracks `m` (self-discovered), not the possibly
    // larger `total_docs` target. Non-resume keeps the original behavior:
    // cluster count from `ingest::n_docs()`.
    let resume_m: Option<usize> = if resume {
        let reader = st.reader().expect("reader");
        let batches = reader
            .query_sql("SELECT COUNT(*) AS n FROM supertable")
            .expect("COUNT(*) on resumed table");
        let m = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("count column is Int64")
            .value(0) as usize;
        Some(m)
    } else {
        None
    };
    // Load the newest persisted GT bin up front (resume only): its header
    // n_cent is the ORIGINAL build's cluster count — provenance a resume cannot
    // re-derive from COUNT(*), since a run that crashed mid-target sits in a
    // different doc-count band than its target. An Incompatible bin
    // (version/dim/k/seeds/n_queries) is a hard error; Corrupt/absent degrades
    // to a replay rebuild.
    let resume_bin: Option<GtBin> = match (resume, &gt_bin_path) {
        (true, Some(base)) => match gt_bin_latest(base) {
            Some((_, path)) => match gt_bin_read(&path, n_queries) {
                Ok(bin) => Some(bin),
                Err(GtBinError::Incompatible(why)) => {
                    panic!("resume: GT bin provenance conflicts with this run: {why}")
                }
                Err(GtBinError::Corrupt(why)) => {
                    eprintln!(
                        "[recall_while_ingest] GT bin unusable ({why}); rebuilding GT by replay"
                    );
                    None
                }
            },
            None => None,
        },
        _ => None,
    };
    // Cluster count for the corpus generator + held-out queries. On resume the
    // bin header is authoritative; without a bin, derive from COUNT(*) — sound
    // only if the prior run FINISHED its target (a crashed cross-band run has no
    // GT to restore anyway).
    let n_cent = match (resume_m, &resume_bin) {
        (Some(m), Some(bin)) => {
            eprintln!(
                "[recall_while_ingest] resume: n_cent={} from GT bin header (provenance) — \
                 COUNT(*)={m} alone would derive {}",
                bin.n_cent,
                corpus::n_cent(m)
            );
            bin.n_cent
        }
        (Some(m), None) => {
            eprintln!(
                "[recall_while_ingest] WARNING: resuming without a usable GT bin; deriving n_cent \
                 from COUNT(*)={m}. Valid only if the prior run finished its target; a crashed \
                 cross-band run needs its GT bin for correct provenance."
            );
            corpus::n_cent(m)
        }
        (None, _) => corpus::n_cent(ingest::n_docs()),
    };

    // Held-out queries + running ground-truth heaps.
    let queries = build_queries(n_cent, n_queries);
    let mut heaps: Vec<HeldTopK> = (0..n_queries).map(|_| HeldTopK::new()).collect();

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

    if let Some(m) = resume_m {
        assert!(
            m <= total_docs,
            "resume: existing table has {m} docs > target {total_docs}; raise INFINO_BENCH_SUPERTABLE_DOCS"
        );
        eprintln!(
            "[recall_while_ingest] RESUME: opened existing table at M={m} docs; \
             restoring ground truth for [0, M) (no re-ingest), then ingesting [{m}, {total_docs})"
        );
        // Ground truth for [0, M): prefer a persisted bin (cheap), else
        // rebuild by replaying the deterministic generator. Either way the
        // generator is advanced over ALL of [0, M) so the subsequent ingest of
        // [M, target) continues the same sequence; only the UNCOVERED portion
        // [gt_covered, M) is scored into the heaps — re-scoring a doc already
        // in a heap could insert a duplicate id and corrupt the top-k.
        // Restore from the bin loaded up front (its n_cent already drove the
        // generator/queries above). Bad/absent bins already degraded to a
        // replay rebuild there; incompatible ones already aborted.
        let mut gt_covered = 0usize;
        match resume_bin {
            Some(bin) if bin.m <= m => {
                eprintln!(
                    "[recall_while_ingest] GT bin covers M={} (index M={m}); query params verified, \
                     bin CONTENTS assumed to match the indexed vectors (not verifiable)",
                    bin.m
                );
                heaps = bin.heaps;
                gt_covered = bin.m;
                if bin.m < m {
                    eprintln!(
                        "[recall_while_ingest] GT bin behind index ({} < {m}); catching up [{}, {m})",
                        bin.m, bin.m
                    );
                }
            }
            Some(bin) => eprintln!(
                "[recall_while_ingest] WARNING: GT bin AHEAD of index ({} > {m}) — recomputing GT \
                 for [0, {m}) (truncating a heap could drop a true neighbor)",
                bin.m
            ),
            None if gt_bin_path.is_some() => eprintln!(
                "[recall_while_ingest] no usable GT bin; building GT by replay, will persist"
            ),
            None => {}
        }
        let mut pos = 0usize;
        while pos < m {
            let sub = MAX_INGEST_BATCH_DOCS.min(m - pos);
            stream.fill_chunk_modality(sub, &mut titles, &mut flat, false, true);
            if pos + sub > gt_covered {
                let skip = gt_covered.saturating_sub(pos);
                update_heaps(
                    &mut heaps,
                    &queries,
                    &flat[skip * DIM..sub * DIM],
                    (pos + skip) as u32,
                    sub - skip,
                );
            }
            pos += sub;
        }
        n = m;
        ingested.store(m, Ordering::Relaxed);

        // Bank the just-restored GT for [0, M) NOW — before the crash-prone
        // ingest begins — so a crash never forces re-replaying it. Skip if a
        // loaded bin already covers exactly M (it's already on disk).
        if gt_covered < m
            && let Some(base) = &gt_bin_path
        {
            let path = gt_bin_ckpt_path(base, m);
            match gt_bin_write(&path, &heaps, m, n_queries, n_cent) {
                Ok(()) => {
                    gt_bin_prune(base, m);
                    eprintln!("[recall_while_ingest] GT bin persisted after replay: {path}");
                }
                Err(e) => eprintln!(
                    "[recall_while_ingest] WARNING: post-replay GT bin persist to {path} failed: {e}"
                ),
            }
        }
    }

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
            let batch = vector_batch(&schema, &flat, sub, n + off);
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

        // Re-open once the first batch is committed. Historically required:
        // a create-era handle's hidden options carried no `VectorCell`
        // strategy and `optimize()` gated the cell-split phase on those
        // options, so only a reopened handle ever split cells. The split
        // gate now keys on the hidden manifest's locked strategy, but the
        // reopen is kept: it pins the measured shape to the steady-state
        // handle (append + optimize + query on one reopened handle), which
        // is what the recorded baselines were taken against.
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
        // Oracle rung (INFINO_BENCH_ORACLE=1): all cells at a rerank budget
        // covering every row — with the width-independent per-cell budget
        // (#537) this exact-scores the whole table through the stored
        // rerank payloads, bypassing 1-bit selection entirely. Recall vs
        // GT here is the engine's stored-payload ceiling (codec error
        // only); the ladder above must approach it as nprobe grows. Costs
        // a full-table rerank per query — gated off by default.
        if cells_now > 0 && std::env::var("INFINO_BENCH_ORACLE").is_ok_and(|v| v == "1") {
            let oracle_rm = n.div_ceil(K).max(1);
            let r = measure_recall_rm(&st, &queries, &heaps, &id_map, cells_now, oracle_rm);
            eprintln!(
                "[breadth-sweep] nprobe={cells_now:<5} recall@10={r:.3} (oracle rm={oracle_rm})"
            );
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

        // Persist the running GT so a crash mid-run resumes cheaply and each
        // checkpoint leaves a reusable oracle covering [0, n). The count is in
        // the filename (`{base}.M{n}.bin`) so `ls` shows how far each covers.
        if let Some(base) = &gt_bin_path {
            let path = gt_bin_ckpt_path(base, n);
            match gt_bin_write(&path, &heaps, n, n_queries, n_cent) {
                Ok(()) => {
                    // Durably written — now the older bins are superseded.
                    gt_bin_prune(base, n);
                    eprintln!("[recall_while_ingest] GT bin persisted: {path} (older bins pruned)");
                }
                Err(e) => {
                    eprintln!("[recall_while_ingest] WARNING: GT bin persist to {path} failed: {e}")
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, unique scratch base path `<tmp>/rwi_gt_<pid>_<tag>/gt`.
    fn tmp_base(tag: &str) -> String {
        let dir = std::env::temp_dir().join(format!("rwi_gt_{}_{tag}", std::process::id()));
        fs::create_dir_all(&dir).expect("mk scratch dir");
        dir.join("gt").to_string_lossy().into_owned()
    }

    fn heap_with(entries: &[(u32, f32)]) -> HeldTopK {
        let mut h = HeldTopK::new();
        for &(id, score) in entries {
            h.offer(score, id);
        }
        h
    }

    /// Heap contents as a sorted `(id, score_bits)` set — order-independent and
    /// bit-exact (the bytes round-trip, so bit equality is the right check).
    fn sorted(h: &HeldTopK) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = h
            .entries()
            .iter()
            .map(|&(id, s)| (id, s.to_bits()))
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn gt_bin_round_trips_header_and_heaps() {
        let path = gt_bin_ckpt_path(&tmp_base("rt"), 12_345);
        let heaps = vec![heap_with(&[(1, 0.9), (2, 0.8)]), heap_with(&[(3, 0.7)])];
        // n_cent=777 is deliberately unrelated to any band — it must survive as
        // provenance, not be re-derived on read.
        gt_bin_write(&path, &heaps, 12_345, heaps.len(), 777).expect("write");
        let bin = gt_bin_read(&path, heaps.len()).expect("read");
        assert_eq!(bin.m, 12_345);
        assert_eq!(bin.n_cent, 777, "n_cent is provenance and must round-trip");
        assert_eq!(bin.heaps.len(), 2);
        assert_eq!(sorted(&bin.heaps[0]), sorted(&heaps[0]));
        assert_eq!(sorted(&bin.heaps[1]), sorted(&heaps[1]));
    }

    #[test]
    fn gt_bin_refuses_mismatched_query_params() {
        let path = gt_bin_ckpt_path(&tmp_base("mm"), 100);
        gt_bin_write(&path, &[heap_with(&[(1, 0.5)])], 100, 1, 64).expect("write");
        // Reading as if this run uses a different n_queries is a provenance
        // conflict — Incompatible (fatal), never a silent rebuild.
        assert!(
            matches!(gt_bin_read(&path, 4), Err(GtBinError::Incompatible(_))),
            "a different n_queries must be refused as Incompatible"
        );
    }

    #[test]
    fn gt_bin_reports_corruption_as_recoverable() {
        let path = gt_bin_ckpt_path(&tmp_base("corrupt"), 1);
        fs::write(&path, b"not a gt bin").expect("write");
        assert!(
            matches!(gt_bin_read(&path, 1), Err(GtBinError::Corrupt(_))),
            "bad magic is Corrupt (replayable), not Incompatible"
        );
    }

    #[test]
    fn gt_bin_latest_picks_highest_count() {
        let base = tmp_base("latest");
        for n in [10usize, 200, 30] {
            fs::write(gt_bin_ckpt_path(&base, n), b"").expect("touch");
        }
        // A non-`.M<digits>.bin` sibling must be ignored, not misparsed.
        fs::write(format!("{base}.Mxx.bin"), b"").expect("touch");
        assert_eq!(gt_bin_latest(&base).map(|(c, _)| c), Some(200));
    }
}
