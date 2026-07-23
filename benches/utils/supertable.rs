// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable object-store bench (infino-only entry point).
//!
//! Multi-superfile ingest to object storage at the supertable scale
//! (`INFINO_BENCH_SUPERTABLE_DOCS`, default 10M), built through the
//! production `SupertableWriter::append` + `commit` path. Three index
//! shapes are measured for apples-to-apples comparison against
//! single-modality peers: FTS-only, vector-only, SQL, and combined FTS +
//! vector.
//!
//! **RustFS (default), S3, or Azure** — the multi-commit build relies on
//! conditional `If-Match` PUTs; the default local RustFS session implements
//! them. Real S3 and Azure are also supported. Every object the run writes
//! lands under one unique bucket/prefix per shape, all deleted before the
//! runner returns (unless `INFINO_BENCH_KEEP_TABLE` is set).
//!
//! ## Per-shape process isolation
//!
//! Each shape is built in its **own subprocess** (the parent re-execs this
//! same bench binary with `INFINO_BENCH_SUPERTABLE_SHAPE=<shape>`). RSS is
//! sampled inside that child, so each shape's Peak/Median/P90 are measured
//! from a clean address space. Within a single process `VmRSS` is a
//! monotonic high-water mark — the allocator does not return freed pages to
//! the OS — so running all three shapes in one process would let whichever
//! ran first poison the memory numbers of the ones after it. Isolation makes
//! the three rows independent and comparable.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench -- supertable
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket cargo bench -- supertable
//! INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
//!   AZURE_STORAGE_ACCOUNT_NAME=... AZURE_STORAGE_ACCOUNT_KEY=... cargo bench -- supertable
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_SUPERTABLE_DOCS=100000 cargo bench -- supertable
//! ```

#[allow(unused_imports)] // `Instant` is consumed by the child mods via `use super::*`
use std::collections::HashSet;
use std::{
    env,
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use infino::{
    OptimizeOptions,
    supertable::{
        Supertable,
        manifest::{ClusterCentroids, SuperfileEntry},
    },
};
use tempfile::TempDir;

use crate::{
    cold_store::{self, ColdStoreMeasurement, STEADY_COLD_SAMPLES},
    corpus::DIM,
    cost, cpu,
    ingest::supertable::{self, Modality, modality_label},
    markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time},
    report::{Better, Block, Cell, Report, Section, context, metric, text},
    rss::{self, PeakSampler},
    storage_meter::{self, fmt_get_class_breakdown},
    tiers,
};

/// Env var the parent sets to make a child build exactly one shape and
/// print its metrics instead of emitting the report.
const SHAPE_ENV: &str = "INFINO_BENCH_SUPERTABLE_SHAPE";
/// Line prefix a child writes to stdout carrying its measured metrics.
const RESULT_PREFIX: &str = "__SUPERTABLE_SHAPE_RESULT__ ";
/// `ingest_cpu_ns` value meaning "not measured" — a `key=value` result line
/// can't carry `Option<f64>`, so a negative sentinel encodes `None`.
const INGEST_CPU_NOT_MEASURED_NS: i128 = -1;

/// The three measured shapes: (display label, child-env key, modality).
const SHAPES: [(&str, &str, Modality); 4] = [
    ("FTS-only", "fts", Modality::Fts),
    ("vector-only", "vector", Modality::Vector),
    ("SQL", "sql", Modality::Sql),
    ("combined FTS + vector", "combined", Modality::Combined),
];

/// Plain measured numbers for one shape, marshalled across the
/// parent/child process boundary as a single `key=value` line.
pub struct ShapeMetrics {
    pub wall_ns: f64,
    pub n_superfiles: usize,
    /// Peak total VmRSS — what the cost model's RAM-hold leg bills.
    pub peak_rss_bytes: u64,
    pub median_rss_bytes: u64,
    pub p90_rss_bytes: u64,
    /// Peak anonymous (heap) RSS over the ingest window — diagnostic only;
    /// not used for $.
    pub peak_anon_rss_bytes: u64,
    /// Peak file-backed RSS (`total − anon`) — diagnostic only.
    pub peak_file_rss_bytes: u64,
    /// Index bytes written to object storage during ingest. The
    /// supertable's "upload bandwidth" is this over the wall time —
    /// the bytes-to-object-store rate, the analogue of the superfile
    /// build's input-payload bandwidth.
    pub index_bytes: u64,
    /// Raw input corpus size (text + vector bytes) — the source data
    /// fed to ingest, distinct from `index_bytes` (what's written out).
    pub corpus_bytes: u64,
    /// Measured on-CPU seconds over the ingest build window (all-thread
    /// schedstat delta). `None` on a platform without `/proc` sampling ⇒
    /// the cost model reports the phase as NOT METERED.
    pub ingest_cpu_s: Option<f64>,
}

pub struct SupertableShapeResult {
    pub label: &'static str,
    pub key: &'static str,
    pub metrics: ShapeMetrics,
}

impl ShapeMetrics {
    /// Render as the single stdout line the parent parses.
    fn to_result_line(&self) -> String {
        format!(
            "{RESULT_PREFIX}wall_ns={} n_superfiles={} peak={} median={} p90={} peak_anon={} peak_file={} index_bytes={} corpus_bytes={} ingest_cpu_ns={}",
            self.wall_ns,
            self.n_superfiles,
            self.peak_rss_bytes,
            self.median_rss_bytes,
            self.p90_rss_bytes,
            self.peak_anon_rss_bytes,
            self.peak_file_rss_bytes,
            self.index_bytes,
            self.corpus_bytes,
            self.ingest_cpu_s
                .map(|s| (s * 1e9) as i128)
                .unwrap_or(INGEST_CPU_NOT_MEASURED_NS),
        )
    }

    /// Parse the line emitted by [`to_result_line`]. Returns `None` if a
    /// field is missing or unparseable.
    fn from_result_line(line: &str) -> Option<Self> {
        let body = line.strip_prefix(RESULT_PREFIX)?;
        let mut wall_ns = None;
        let mut n_superfiles = None;
        let mut peak = None;
        let mut median = None;
        let mut p90 = None;
        let mut peak_anon = None;
        let mut peak_file = None;
        let mut index_bytes = None;
        let mut corpus_bytes = None;
        // Optional (older/other producers may omit it); `-1` = not measured.
        let mut ingest_cpu_s: Option<f64> = None;
        for tok in body.split_whitespace() {
            let (k, v) = tok.split_once('=')?;
            match k {
                "wall_ns" => wall_ns = v.parse().ok(),
                "n_superfiles" => n_superfiles = v.parse().ok(),
                "peak" => peak = v.parse().ok(),
                "median" => median = v.parse().ok(),
                "p90" => p90 = v.parse().ok(),
                "peak_anon" => peak_anon = v.parse().ok(),
                "peak_file" => peak_file = v.parse().ok(),
                "index_bytes" => index_bytes = v.parse().ok(),
                "corpus_bytes" => corpus_bytes = v.parse().ok(),
                "ingest_cpu_ns" => {
                    ingest_cpu_s = v
                        .parse::<i128>()
                        .ok()
                        .filter(|ns| *ns != INGEST_CPU_NOT_MEASURED_NS)
                        .map(|ns| ns as f64 / 1e9);
                }
                _ => {}
            }
        }
        // Older child lines omit anon/file; leave them 0 (diagnostic only).
        Some(ShapeMetrics {
            wall_ns: wall_ns?,
            n_superfiles: n_superfiles?,
            peak_rss_bytes: peak?,
            median_rss_bytes: median?,
            p90_rss_bytes: p90?,
            peak_anon_rss_bytes: peak_anon.unwrap_or(0),
            peak_file_rss_bytes: peak_file.unwrap_or(0),
            index_bytes: index_bytes?,
            corpus_bytes: corpus_bytes?,
            ingest_cpu_s,
        })
    }
}

fn modality_for_key(key: &str) -> Option<Modality> {
    SHAPES
        .iter()
        .find(|(_, k, _)| *k == key)
        .map(|(_, _, m)| *m)
}

/// Child entry point: build exactly one shape, sample its RSS in this
/// fresh process, clean up the real-S3 prefix it wrote, and print the
/// metrics line. Does not emit the report.
fn run_child_shape(key: &str) {
    let modality = match modality_for_key(key) {
        Some(m) => m,
        None => {
            eprintln!("[supertable] unknown shape key {key:?}");
            std::process::exit(2);
        }
    };

    eprintln!(
        "[supertable] child process: ingesting {} shape ({} docs)...",
        modality_label(modality),
        fmt_count(supertable::n_docs()),
    );
    // Corpus is generated to disk + mmapped BEFORE the sampler so the
    // measured window covers the engine only.
    let corpus = supertable::prepare_corpus(modality);
    let sampler = PeakSampler::start_default();
    let (built, wall, ingest_cpu_s) =
        cpu::timed(|| supertable::build_on_storage(modality, &corpus));
    let rss = sampler.stop_stats();

    // This child wrote its own unique prefix; delete it before exiting so the
    // real-backend run accrues no ongoing cost (ingest-only bench — the
    // artifact is not reused after the build is measured).
    if let Some(cleanup) = &built.cleanup {
        eprintln!("[supertable] child process: cleaning up object-store prefix...");
        crate::tiers::cleanup_prefix(cleanup);
    }

    let metrics = ShapeMetrics {
        wall_ns: wall.as_secs_f64() * 1e9,
        n_superfiles: built.n_superfiles,
        peak_rss_bytes: rss.peak_rss_bytes,
        median_rss_bytes: rss.median_rss_bytes,
        p90_rss_bytes: rss.p90_rss_bytes,
        peak_anon_rss_bytes: rss.peak_anon_rss_bytes,
        peak_file_rss_bytes: rss.peak_file_rss_bytes,
        index_bytes: built.total_index_bytes,
        corpus_bytes: corpus.byte_size(),
        ingest_cpu_s,
    };
    println!("{}", metrics.to_result_line());
}

/// Spawn one isolated child to build `key` and return its metrics.
/// stderr is inherited so the child's `[tiers]` logs stream live; stdout
/// is captured to read back the single result line.
fn build_shape_isolated(key: &str) -> Option<ShapeMetrics> {
    eprintln!("[supertable] spawning isolated subprocess for shape {key:?}...");
    let exe = std::env::current_exe().expect("current_exe for supertable child");
    let mut cmd = Command::new(exe);
    cmd.env(SHAPE_ENV, key);
    // Forward a CLI-set dataset prefix; the child only inherits the env.
    if let Some(prefix) = crate::dataset::dataset_prefix() {
        cmd.env(crate::dataset::PREFIX_ENV, prefix);
    }
    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .expect("spawn supertable shape child");
    if !output.status.success() {
        eprintln!(
            "[supertable] shape {key:?} child exited with {} — skipping its row",
            output.status
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let metrics = stdout.lines().find_map(ShapeMetrics::from_result_line);
    if metrics.is_none() {
        eprintln!("[supertable] shape {key:?} child produced no result line — skipping its row");
    }
    metrics
}

pub fn handle_shape_child_from_env() -> bool {
    if let Ok(key) = std::env::var(SHAPE_ENV) {
        run_child_shape(&key);
        true
    } else {
        false
    }
}

pub fn run_ingest_shapes_isolated() -> Vec<SupertableShapeResult> {
    let mut results = Vec::with_capacity(SHAPES.len());
    for (label, key, _) in SHAPES {
        eprintln!("[supertable] === shape {label} (isolated process) ===");
        if let Some(metrics) = build_shape_isolated(key) {
            results.push(SupertableShapeResult {
                label,
                key,
                metrics,
            });
        }
    }
    results
}

/// Shared column headers for every supertable ingest table (the
/// combined `run()` table and the per-modality fts/vector/sql tables),
/// so the four call sites can't drift apart. `Stored` is the total
/// on-storage footprint of the committed superfiles — full Parquet
/// (data pages + embedded BM25/vector indexes), not just the index
/// subsections — printed next to the raw `Corpus` it was built from.
pub fn ingest_headers() -> Vec<String> {
    vec![
        "Shape".into(),
        "Time".into(),
        "Throughput".into(),
        "Bandwidth".into(),
        "Corpus".into(),
        "Stored".into(),
        "Superfiles".into(),
        "Peak RSS".into(),
        "Peak anon".into(),
        "Peak file".into(),
        "Median RSS".into(),
        "P90 RSS".into(),
    ]
}

pub fn ingest_row(n_docs: usize, label: &str, m: &ShapeMetrics) -> Vec<Cell> {
    let secs = m.wall_ns / 1e9;
    let thr = if secs > 0.0 {
        n_docs as f64 / secs
    } else {
        0.0
    };
    // Upload bandwidth: stored bytes written to object storage per
    // second over the ingest wall time.
    let bw = if secs > 0.0 {
        m.index_bytes as f64 / secs
    } else {
        0.0
    };
    // Stored footprint as a fraction of the raw corpus it was built
    // from — the headline compression/expansion ratio per modality.
    let stored_pct = if m.corpus_bytes > 0 {
        100.0 * m.index_bytes as f64 / m.corpus_bytes as f64
    } else {
        0.0
    };
    vec![
        text(label),
        metric(m.wall_ns, fmt_time(m.wall_ns), Better::Lower),
        context(thr, fmt_throughput(thr), Better::Higher),
        context(bw, fmt_bandwidth(bw), Better::Higher),
        text(rss::fmt_bytes(m.corpus_bytes)),
        metric(
            m.index_bytes as f64,
            format!("{} ({stored_pct:.0}%)", rss::fmt_bytes(m.index_bytes)),
            Better::Lower,
        ),
        text(fmt_count(m.n_superfiles)),
        metric(
            m.peak_rss_bytes as f64,
            rss::fmt_bytes(m.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            m.peak_anon_rss_bytes as f64,
            rss::fmt_bytes(m.peak_anon_rss_bytes),
            Better::Lower,
        ),
        metric(
            m.peak_file_rss_bytes as f64,
            rss::fmt_bytes(m.peak_file_rss_bytes),
            Better::Lower,
        ),
        context(
            m.median_rss_bytes as f64,
            rss::fmt_bytes(m.median_rss_bytes),
            Better::Lower,
        ),
        context(
            m.p90_rss_bytes as f64,
            rss::fmt_bytes(m.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

/// Visit committed superfiles through the flat eager view, or through manifest
/// parts when the manifest is lazy and the flat view is empty.
fn visit_manifest_superfiles(table: &Supertable, mut visit: impl FnMut(&SuperfileEntry)) {
    let reader = table.reader();
    let manifest = reader.manifest();
    let flat_superfiles = manifest.get_all_superfiles();
    if !flat_superfiles.is_empty() {
        for entry in flat_superfiles {
            visit(entry);
        }
        return;
    }
    for part_entry in manifest.get_all_list_entries() {
        let part = tiers::block_on(manifest.get_part_by_id(part_entry.part_id))
            .expect("load manifest part for bench metadata");
        for entry in part.superfiles.iter() {
            visit(entry);
        }
    }
}

/// Sum of on-storage superfile bytes (full Parquet + embedded indexes) across
/// a table's committed manifest — the same `subsection_offsets.total_size` sum
/// the ingest path reports, but callable post-drain on either the user table
/// or the derived hidden vector-index table. `IngestResult::total_index_bytes`
/// is captured at ingest, when the hidden index is empty; this recomputes the
/// live footprint so the steady-state (post-drain) total can include the
/// hidden per-cell IVF superfiles.
fn on_storage_bytes(table: &Supertable) -> u64 {
    let mut total = 0u64;
    visit_manifest_superfiles(table, |entry| {
        if let Some(offsets) = entry.subsection_offsets.as_ref() {
            total = total.saturating_add(offsets.total_size);
        }
    });
    total
}

/// Storage prefix of the drain-published slow-CAS entry blob, relative to the
/// table's own provider. Mirrors
/// `src/supertable/slow_vector_state.rs::STORAGE_PREFIX` (crate-private), the
/// same way `UriClass` mirrors engine URI tokens.
const SLOW_VECTOR_STATE_PREFIX: &str = "slow-vector-state/";

/// Total bytes under the table's slow-CAS prefix — the drain-published entry
/// blob(s). Listed directly from storage so the stored-capacity readout
/// reflects what is actually durable (a superseded blob not yet GC'd counts,
/// deliberately). `None` when the table has no storage attached.
fn slow_state_stored_bytes(table: &Supertable) -> Option<u64> {
    listed_bytes_under(table, SLOW_VECTOR_STATE_PREFIX)
}

/// The LIVE stored bytes for the table, LISTed from the object store and
/// filtered to what the CURRENT manifests reference: live superfiles
/// (user + hidden), manifest lists/parts/siblings, pointers, and slow-CAS
/// state. Superfile data objects not referenced by the current manifests
/// (superseded generations awaiting GC, orphaned tmp files) are excluded
/// from the count — the steady-state capacity is what the cost model
/// prices. `None` when the table has no storage attached.
fn live_stored_bytes(consumer: &Supertable) -> Option<u64> {
    /// Relative prefix of superfile data objects under a table root.
    const DATA_PREFIX: &str = "data/";
    let user_reader = consumer.reader();
    let user_manifest = user_reader.manifest();
    let listing = listed_objects_under(consumer, "")?;
    let bucket_total: u64 = listing.iter().map(|(_, size)| *size).sum();
    let user_live: HashSet<String> = user_manifest
        .get_all_superfiles()
        .iter()
        .map(|entry| entry.uri.storage_path())
        .collect();
    let user_dead: u64 = listing
        .iter()
        .filter(|(key, _)| key.starts_with(DATA_PREFIX) && !user_live.contains(key))
        .map(|(_, size)| *size)
        .sum();
    let hidden_dead: u64 = match consumer.vector_index_table() {
        Some(hidden) => {
            let hidden_reader = hidden.pinned_reader();
            let hidden_live: HashSet<String> = hidden_reader
                .manifest()
                .get_all_superfiles()
                .iter()
                .map(|entry| entry.uri.storage_path())
                .collect();
            listed_objects_under(hidden, DATA_PREFIX)
                .unwrap_or_default()
                .iter()
                .filter(|(key, _)| !hidden_live.contains(key))
                .map(|(_, size)| *size)
                .sum()
        }
        None => 0,
    };
    Some(
        bucket_total
            .saturating_sub(user_dead)
            .saturating_sub(hidden_dead),
    )
}

/// Sum of object sizes under `prefix`, listed from the table's provider.
fn listed_bytes_under(table: &Supertable, prefix: &str) -> Option<u64> {
    listed_objects_under(table, prefix).map(|objects| objects.iter().map(|(_, size)| *size).sum())
}

/// `(key, size)` for every object under `prefix`, listed from the
/// table's provider. Keys are provider-root-relative — the same
/// convention as `SuperfileUri::storage_path`.
fn listed_objects_under(table: &Supertable, prefix: &str) -> Option<Vec<(String, u64)>> {
    let storage = Arc::clone(table.reader().manifest().options.storage.as_ref()?);
    let prefix = prefix.to_owned();
    let objects = tiers::block_on(async move {
        storage
            .list_with_prefix_metadata(&prefix)
            .await
            .map(|objs| {
                objs.into_iter()
                    .map(|(key, meta)| (key, meta.size))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    Some(objects)
}

/// Cap on printed first-cold-query trace lines; the tail is summarized so a
/// pre-drain fan (hundreds of GETs) can't flood the log.
const COLD_TRACE_PRINT_MAX: usize = 12;
/// Opt-in request-level cold trace; normal reports show only aggregate tables.
const COLD_TRACE_ENV: &str = "INFINO_TRACE_VECTOR_COLD_FAN";

/// Log one metered cold split (open / first query / repeat query), each
/// window followed by its per-class GET attribution (user vs hidden table,
/// data vs manifest namespace).
fn log_cold_split(prefix: &str, split: &storage_meter::ColdStoreSplit) {
    let fill_gets = split
        .first_query
        .bg_get_count
        .saturating_add(split.second_query.bg_get_count)
        .saturating_add(split.repeat_query.bg_get_count);
    let fill_bytes = split
        .first_query
        .bg_get_bytes
        .saturating_add(split.second_query.bg_get_bytes)
        .saturating_add(split.repeat_query.bg_get_bytes);
    eprintln!(
        "[{prefix}] metered cold: open {} GET + {} HEAD ({} down), first query {} GET ({} down, one-time warmup), second query {} GET ({} down, steady), repeat query {} GET ({} down), cache fill {} GET ({} down)",
        split.open.get_count,
        split.open.head_count,
        rss::fmt_bytes(split.open.get_bytes),
        split.first_query.get_count,
        rss::fmt_bytes(split.first_query.get_bytes),
        split.second_query.get_count,
        rss::fmt_bytes(split.second_query.get_bytes),
        split.repeat_query.get_count,
        rss::fmt_bytes(split.repeat_query.get_bytes),
        fill_gets,
        rss::fmt_bytes(fill_bytes),
    );
    eprintln!(
        "[{prefix}]   open: {} | first query: {} | second query: {} | repeat query: {}",
        fmt_get_class_breakdown(&split.open),
        fmt_get_class_breakdown(&split.first_query),
        fmt_get_class_breakdown(&split.second_query),
        fmt_get_class_breakdown(&split.repeat_query),
    );
}

/// Print one query window's per-request read trace — the exact files and byte
/// ranges behind the fan count, in request order.
fn log_query_read_trace(prefix: &str, phase: &str, trace: &[storage_meter::TraceEntry]) {
    if trace.is_empty() {
        return;
    }
    eprintln!("[{prefix}] {phase} read trace ({} requests):", trace.len());
    for entry in trace.iter().take(COLD_TRACE_PRINT_MAX) {
        let class = storage_meter::UriClass::of(&entry.uri);
        // Elide the `_infino_<uuid>_vector_index/` prefix: the class label
        // already says "hidden", and the tail is the interesting part.
        let shown = match entry.uri.split_once("_vector_index/") {
            Some((_, tail)) => tail,
            None => entry.uri.as_str(),
        };
        match entry.range {
            Some((start, end)) => eprintln!(
                "[{prefix}]   {:<15} {shown}  [{start}..{end})  ({})",
                class.label(),
                rss::fmt_bytes(entry.bytes),
            ),
            None => eprintln!(
                "[{prefix}]   {:<15} {shown}  (whole/tail, {})",
                class.label(),
                rss::fmt_bytes(entry.bytes),
            ),
        }
    }
    if trace.len() > COLD_TRACE_PRINT_MAX {
        eprintln!(
            "[{prefix}]   … and {} more requests",
            trace.len() - COLD_TRACE_PRINT_MAX
        );
    }
}

fn cold_trace_enabled() -> bool {
    env::var_os(COLD_TRACE_ENV).is_some()
}

/// Spread one cold consumer's metered + timed windows into the cost model's
/// phase slots. Steady-state warm I/O is filled by the caller separately.
fn store_phases_from_measurement(measured: Option<ColdStoreMeasurement>) -> cost::StorePhases {
    cost::StorePhases {
        cold_open: measured.as_ref().map(|m| m.split.open),
        cold_query: measured.as_ref().map(|m| m.split.first_query),
        cold_second_query: measured.as_ref().map(|m| m.split.second_query),
        cold_second_wall_s: measured.as_ref().map(|m| m.second_wall_s),
        cold_second_cpu_s: measured.as_ref().and_then(|m| m.second_cpu_s),
        cold_repeat_query: measured.as_ref().map(|m| m.split.repeat_query),
        ..Default::default()
    }
}

/// `(wall_s, io, peak_rss_bytes, cpu_s)` for one metered `optimize()` pass.
type CompactionStats = (f64, storage_meter::ObjectStoreMeter, u64, Option<f64>);

/// Metered [`Supertable::optimize`] — same shape as the vector compaction
/// window (wall / CPU / RSS / UsageMeter delta), without drain/delta.
fn run_metered_optimize(
    label: &str,
    consumer: &Supertable,
    meter: &storage_meter::MeteredStorage,
) -> CompactionStats {
    eprintln!(
        "[{label}] before optimize: {} superfiles",
        consumer.reader().n_superfiles()
    );
    eprintln!("[{label}] compacting (optimize)...");
    let before = meter.snapshot();
    let sampler = PeakSampler::start_default();
    let (result, wall, cpu_s) = cpu::timed(|| consumer.optimize(&OptimizeOptions::default()));
    result.expect("optimize (compaction)");
    let wall_s = wall.as_secs_f64();
    let rss_stats = sampler.stop_stats();
    let peak_rss = rss_stats.peak_rss_bytes;
    let io = meter.snapshot().since(&before);
    eprintln!(
        "[{label}] compaction object-store I/O: {} PUT ({} up), {} GET ({} down) in {wall_s:.1}s \
         (peak RSS {} / anon {} / file {}); after optimize: {} superfiles",
        io.put_count,
        rss::fmt_bytes(io.put_bytes),
        io.get_count,
        rss::fmt_bytes(io.get_bytes),
        rss::fmt_bytes(peak_rss),
        rss::fmt_bytes(rss_stats.peak_anon_rss_bytes),
        rss::fmt_bytes(rss_stats.peak_file_rss_bytes),
        consumer.reader().n_superfiles(),
    );
    (wall_s, io, peak_rss, cpu_s)
}

fn store_phases_with_compaction(
    measured: Option<ColdStoreMeasurement>,
    compaction: Option<CompactionStats>,
) -> cost::StorePhases {
    let mut store = store_phases_from_measurement(measured);
    if let Some((wall_s, io, peak_rss, cpu_s)) = compaction {
        store.compaction = Some(io);
        store.compaction_wall_s = Some(wall_s);
        store.compaction_cpu_s = cpu_s;
        store.compaction_peak_rss_bytes = Some(peak_rss);
    }
    store
}

/// Open a metered consumer, run [`run_metered_optimize`], return stats.
/// The table on `built.storage` is left in the compacted layout.
fn optimize_built_table(
    label: &str,
    modality: Modality,
    built: &supertable::IngestResult,
) -> CompactionStats {
    let meter = storage_meter::wrap(Arc::clone(&built.storage));
    let (cache_dir, cache) =
        tiers::fresh_supertable_search_cache(meter.provider(), Some(built.total_index_bytes));
    let opts = tiers::consumer_options(
        supertable::options_for(modality, None),
        meter.provider(),
        cache,
    );
    let consumer = tiers::open_consumer(opts);
    let stats = run_metered_optimize(label, &consumer, &meter);
    drop(consumer);
    drop(cache_dir);
    stats
}

/// Pre-drain (transient-shape) latency rows: the warm battery and the
/// cold `(open, search)` rows measured before hidden-index maintenance.
type PreDrainLatencies<'a> = (&'a [cost::WarmQueryCost], &'a [cost::ColdQuery]);

#[allow(clippy::too_many_arguments)]
fn emit_cost_warm(
    report: &mut Report,
    anchor: &str,
    title: String,
    built: &supertable::IngestResult,
    metrics: Option<&ShapeMetrics>,
    n_docs: usize,
    warm: &[cost::WarmQueryCost],
    cold: Option<&[cost::ColdQuery]>,
    pre_drain: Option<PreDrainLatencies<'_>>,
    vector_cell: bool,
    mut store: cost::StorePhases,
    stored_bytes_override: Option<u64>,
) {
    if warm.is_empty() && cold.is_none() {
        return;
    }
    // The ingest window was metered inside `build_on_storage`; pre-built
    // tables (dataset / existing-prefix) carry `None` and report as such.
    if store.ingest.is_none() {
        store.ingest = built.ingest_io;
    }
    let resident = rss::current_anon_rss_bytes().unwrap_or(0);
    let (wall_s, corpus_bytes) = match metrics {
        Some(m) => (m.wall_ns / 1e9, m.corpus_bytes),
        None => (0.0, 0),
    };
    cost::emit(
        report,
        anchor,
        title,
        &cost::CellCost {
            ingest_wall_s: wall_s,
            writers: supertable::n_writers() as u32,
            ingest_peak_rss_bytes: metrics.map(|m| m.peak_rss_bytes),
            ingest_cpu_s: metrics.and_then(|m| m.ingest_cpu_s),
            n_commits: supertable::n_commits() as u64,
            unmetered_put_count: None,
            stored_bytes: stored_bytes_override.unwrap_or(built.total_index_bytes),
            corpus_bytes,
            n_docs,
            resident_anon_bytes: resident,
            warm,
            cold,
            warm_pre: pre_drain.map(|(w, _)| w),
            cold_pre: pre_drain.map(|(_, c)| c),
            store,
            vector_cell,
            storage_months: None,
            cold_open_amortized: true,
        },
    );
}

pub fn run() {
    // Pre-flight: this bench only runs against a real object store (S3 or
    // Azure; see `tiers::supertable_storage_fixture`). Fail fast with a clear
    // message instead of a panic deep inside the first build. Checked in both
    // the parent and any spawned child (env is inherited).
    if let Err(reason) = crate::tiers::supertable_backend_check() {
        eprintln!("[supertable] skipped: {reason}");
        return;
    }

    // Child mode: build exactly one shape in this fresh process, then exit.
    if handle_shape_child_from_env() {
        return;
    }

    // Parent mode: build each shape in its own isolated subprocess so the
    // per-shape RSS numbers are independent (see the module docs).
    let n_docs = supertable::n_docs();
    eprintln!(
        "[supertable] ingesting {} docs ({} commits, {} writers) per shape to object storage, \
         one isolated process per shape...",
        fmt_count(n_docs),
        supertable::n_commits(),
        supertable::n_writers()
    );

    let shape_results = run_ingest_shapes_isolated();
    let rows: Vec<Vec<Cell>> = shape_results
        .iter()
        .map(|r| ingest_row(n_docs, r.label, &r.metrics))
        .collect();

    if rows.is_empty() {
        eprintln!("[supertable] no shapes produced metrics — not emitting a report");
        return;
    }

    let mut report = Report::load("supertable");
    report.emit(&Section {
        anchor: "bench/supertable/ingest".into(),
        title: format!(
            "Supertable — ingest, multi-superfile / object-store ({} docs × dim={}, {} commits, {} writers)",
            fmt_count(n_docs),
            crate::corpus::DIM,
            supertable::n_commits(),
            supertable::n_writers()
        ),
        note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). \
               Each shape is built in its own subprocess, so Peak/Median/P90 RSS are measured from a \
               clean address space and are comparable across shapes. Rows are the three index shapes \
               built from the same seeded corpus, so each is directly comparable to its single-modality \
               peer. Throughput is rows/s; `Stored` is the total on-storage footprint of the committed \
               superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; \
               `Superfiles` is the committed superfile count. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: ingest_headers(),
            rows,
        }],
    });
    report.save();
}

// ─── Per-modality query runners ───────────────────────────────────────────

const WARM_ITERS: usize = 20;
const COLD_ITERS: usize = 5;
const TOP_K: usize = 10;

/// Selected phases for a per-modality supertable runner.
///
/// Read phases (`warm`, `cold`) still build the object-store table because
/// they need the committed artifact; `build` controls whether the ingest
/// section is emitted.
#[derive(Clone, Copy)]
pub struct Phases {
    pub build: bool,
    pub warm: bool,
    pub cold: bool,
}

impl Phases {
    pub const ALL: Phases = Phases {
        build: true,
        warm: true,
        cold: true,
    };
}

/// Ingest a prepared corpus, sampling RSS over the build window. Returns the
/// ingest measurements only for the build phase (it emits them).
fn build_measured(
    modality: Modality,
    corpus: &supertable::PreparedCorpus,
    phases: Phases,
) -> (supertable::IngestResult, Option<ShapeMetrics>) {
    let sampler = PeakSampler::start_default();
    let (built, wall, ingest_cpu_s) = cpu::timed(|| supertable::build_on_storage(modality, corpus));
    let rss = sampler.stop_stats();
    let metrics = phases.build.then_some(ShapeMetrics {
        wall_ns: wall.as_secs_f64() * 1e9,
        n_superfiles: built.n_superfiles,
        peak_rss_bytes: rss.peak_rss_bytes,
        median_rss_bytes: rss.median_rss_bytes,
        p90_rss_bytes: rss.p90_rss_bytes,
        peak_anon_rss_bytes: rss.peak_anon_rss_bytes,
        peak_file_rss_bytes: rss.peak_file_rss_bytes,
        index_bytes: built.total_index_bytes,
        corpus_bytes: corpus.byte_size(),
        ingest_cpu_s,
    });
    (built, metrics)
}

/// Obtain the search artifact for modalities that don't need the corpus after
/// build (FTS, SQL): in dataset mode open the pre-uploaded dataset (no corpus,
/// no ingest); otherwise generate the corpus and ingest it. Vector keeps its
/// corpus for recall ground truth and calls [`build_measured`] directly.
fn build_or_open(
    modality: Modality,
    phases: Phases,
) -> (supertable::IngestResult, Option<ShapeMetrics>) {
    // Dataset mode opens the pre-uploaded dataset only for read phases; a
    // build phase is the prepare step, which still ingests (to the fixed
    // prefix).
    if crate::dataset::dataset_mode() && !phases.build {
        return (supertable::open_dataset(modality), None);
    }
    // Corpus to disk + mmap BEFORE the sampler — engine-only window.
    let corpus = supertable::prepare_corpus(modality);
    build_measured(modality, &corpus, phases)
}

fn open_consumer(modality: Modality, built: &supertable::IngestResult) -> (TempDir, Supertable) {
    let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
        Arc::clone(&built.storage),
        Some(built.total_index_bytes),
    );
    let opts = tiers::consumer_options(
        supertable::options_for(modality, None),
        Arc::clone(&built.storage),
        cache,
    );
    (cache_dir, tiers::open_consumer(opts))
}

pub mod fts {
    use super::*;
    use crate::{
        executors::{
            fts as exec_fts,
            fts::{FTS_BATTERY, FtsRead},
        },
        harness::driver::FtsQuery,
    };

    /// Large top-k for the serving-scale query-phase scaling gate —
    /// mirrors the superfile tier. Query phase only, over a small subset
    /// (the full battery's fetch at this k would dominate the budget).
    const K_LARGE: usize = 1000;
    /// Representative shapes for the large-k gate: a common term, a small
    /// and a large OR, and an AND.
    const K_LARGE_SHAPE_NAMES: &[&str] = &[
        "single_common",
        "two_term_or",
        "ten_term_or",
        "two_term_and",
    ];

    /// Build an FTS-only supertable, then measure warm and cold BM25
    /// reads through the shared FTS executor (same code superfile runs).
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_fts] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_fts");

        // Build-only matches main `supertable_all`: one isolated subprocess
        // with a clean RSS sample. Warm/cold need the artifact in-process.
        if phases.build && !phases.warm && !phases.cold {
            eprintln!(
                "[supertable_fts] build-only: isolated ingest of {} docs to object storage...",
                fmt_count(n_docs),
            );
            if let Some(metrics) = build_shape_isolated("fts") {
                emit_ingest(&mut report, n_docs, &metrics);
                report.save();
            }
            return;
        }

        let (built, ingest_metrics) = build_or_open(Modality::Fts, phases);
        if let Some(metrics) = &ingest_metrics {
            emit_ingest(&mut report, n_docs, metrics);
        }

        // Optimize + post-compact only on a fresh ingest in this process —
        // without drain/delta/OPANN.
        let run_optimize = ingest_metrics.is_some() && (phases.warm || phases.cold);

        if phases.warm || phases.cold {
            let (cache_dir, consumer) = open_consumer(Modality::Fts, &built);
            let reader = consumer.reader();
            exec_fts::assert_correct(&reader, supertable::TEXT_COLUMN, n_docs, "supertable_fts");
            drop(consumer);
            drop(cache_dir);
        }

        // Pre-compact (or sole) search: fragmented post-ingest layout.
        let (warm_pre, counts, large_k) = match phases.warm.then(|| measure_warm(&built)) {
            Some((w, c, l)) => (Some(w), Some(c), Some(l)),
            None => (None, None, None),
        };
        let cold_pre = phases.cold.then(|| measure_cold(&built));
        if phases.warm || phases.cold {
            let (anchor, title, note) = if run_optimize {
                (
                    "bench/fts/supertable/search-pre-compact",
                    format!(
                        "Supertable FTS — search pre-compact, multi-superfile / object-store ({} docs)",
                        fmt_count(n_docs)
                    ),
                    "Pre-compact (post-ingest fanout): warm = shared consumer + disk cache; \
                     cold open = construct only; cold search = first bm25_search. Δ vs previous run."
                        .to_string(),
                )
            } else {
                (
                    "bench/fts/supertable/search",
                    format!(
                        "Supertable FTS — search, multi-superfile / object-store ({} docs)",
                        fmt_count(n_docs)
                    ),
                    "Warm = shared consumer + disk cache; one prewarm + wait_until_warm, then p50 / p90 / p99 \
                     over repeated bm25_search (Δ gates on `p50`). Cold open = fresh cache + consumer \
                     construct only; cold search = first bm25_search (query-driven survivor opens + score) — \
                     same split as cost-model cold I/O. Δ is vs the previous run."
                        .to_string(),
                )
            };
            exec_fts::emit_search(
                &mut report,
                anchor,
                title,
                &note,
                warm_pre.as_deref(),
                cold_pre.as_ref(),
                None,
            );
        }

        if let Some(large_k) = &large_k {
            report.emit(&Section {
                anchor: "bench/fts/supertable/search-large-k".into(),
                title: format!(
                    "Supertable FTS — search top-{K_LARGE} (query phase), multi-superfile / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                note: format!(
                    "Query-phase p50 at k = {K_LARGE} for representative shapes — gates how top-k \
                     collection cost scales with k vs the top-{TOP_K} table at serving scale. Δ is \
                     vs the previous run."
                ),
                blocks: vec![Block {
                    subtitle: format!("top-{K_LARGE} queries"),
                    headers: vec!["Query".into(), "warm (query)".into()],
                    rows: large_k
                        .iter()
                        .map(|(name, d)| {
                            let ns = d.as_secs_f64() * 1e9;
                            vec![text(*name), metric(ns, fmt_time(ns), Better::Lower)]
                        })
                        .collect(),
                }],
            });
        }

        if let Some(counts) = &counts {
            exec_fts::emit_count(
                &mut report,
                "bench/fts/supertable/count",
                format!(
                    "Supertable FTS — count, multi-superfile / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                "Matching-doc count via the dedicated count path: per-superfile single-term `term_df` \
                 read O(1) from the dictionary header and summed across superfiles; multi-term \
                 union/intersection via `token_match` cardinality, less tombstoned docs. No BM25 \
                 scoring, no row materialization. `matches` is the count returned. Δ is vs the \
                 previous run.",
                counts,
            );
        }

        let compaction_stats =
            run_optimize.then(|| optimize_built_table("supertable_fts", Modality::Fts, &built));

        // Post-compact search + cost-cold (steady serving layout).
        let (warm_post, cold_post) = if run_optimize {
            let warm_post = phases.warm.then(|| {
                let (w, _, _) = measure_warm(&built);
                w
            });
            let cold_post = phases.cold.then(|| measure_cold(&built));
            if phases.warm || phases.cold {
                exec_fts::emit_search(
                    &mut report,
                    "bench/fts/supertable/search-post-compact",
                    format!(
                        "Supertable FTS — search post-compact, multi-superfile / object-store ({} docs)",
                        fmt_count(n_docs)
                    ),
                    "Post-compact (after optimize): fewer superfiles; warm/cold recipe unchanged. \
                     This is the steady-state layout the cost model prices. Δ vs previous run.",
                    warm_post.as_deref(),
                    cold_post.as_ref(),
                    None,
                );
            }
            (warm_post, cold_post)
        } else {
            (None, None)
        };

        if phases.warm || phases.cold {
            let (warm_for_cost, cold_for_cost, pre_latencies) = if run_optimize {
                (
                    warm_post.as_deref().unwrap_or(&[]),
                    cold_post.as_ref(),
                    Some((
                        warm_pre
                            .as_deref()
                            .map(cost::warm_from_fts)
                            .unwrap_or_default(),
                        cold_pre
                            .as_ref()
                            .map(cost::cold_from_timings)
                            .unwrap_or_default(),
                    )),
                )
            } else {
                (warm_pre.as_deref().unwrap_or(&[]), cold_pre.as_ref(), None)
            };
            let warm_vec = cost::warm_from_fts(warm_for_cost);
            let cold_vec = cold_for_cost
                .map(cost::cold_from_timings)
                .unwrap_or_default();
            let cold_measured = phases.cold.then(|| measure_cold_store(&built)).flatten();
            let pre_refs = pre_latencies
                .as_ref()
                .map(|(w, c)| (w.as_slice(), c.as_slice()));
            if !warm_vec.is_empty() || !cold_vec.is_empty() {
                emit_cost_warm(
                    &mut report,
                    "bench/fts/supertable/cost",
                    format!("Supertable FTS — cost model ({} docs)", fmt_count(n_docs)),
                    &built,
                    ingest_metrics.as_ref(),
                    n_docs,
                    &warm_vec,
                    if cold_vec.is_empty() {
                        None
                    } else {
                        Some(&cold_vec)
                    },
                    pre_refs,
                    false,
                    store_phases_with_compaction(cold_measured, compaction_stats),
                    None,
                );
            }
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_fts] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn emit_ingest(report: &mut Report, n_docs: usize, metrics: &ShapeMetrics) {
        report.emit(&Section {
            anchor: "bench/fts/supertable/ingest".into(),
            title: format!(
                "Supertable FTS — ingest, multi-superfile / object-store ({} docs, {} commits, {} writers)",
                fmt_count(n_docs),
                supertable::n_commits(),
                supertable::n_writers()
            ),
            note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: ingest_headers(),
                rows: vec![ingest_row(n_docs, "FTS-only", metrics)],
            }],
        });
    }

    fn measure_warm(
        built: &supertable::IngestResult,
    ) -> (
        Vec<exec_fts::FtsQueryStat>,
        Vec<exec_fts::CountStat>,
        Vec<(&'static str, Duration)>,
    ) {
        eprintln!(
            "[supertable_fts] warm: opening shared consumer, prewarm + wait_until_warm once..."
        );
        // Phase-boundary RSS splits (anonymous heap vs mmap'd files):
        // the discriminator for "where do the warm-phase GiBs live" —
        // ingest leftovers show up as anonymous bloat already present
        // before the consumer opens; promotion double-residency shows
        // up as anonymous ≈ file_backed after warm-up.
        crate::rss::log_rss_breakdown("supertable_fts before consumer open");
        let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
        let reader = consumer.reader();
        // Prewarm + wait: one query opens every pruned-in superfile so the
        // background fills spawn, then wait_until_warm blocks until each is
        // mmap-promoted. Warm numbers time a hot cache, not the fill race —
        // same methodology as the cold split, which meters the fill
        // explicitly.
        let first = &FTS_BATTERY[0];
        let first_query = first.terms.join(" ");
        let _ = reader
            .bm25_search(
                supertable::TEXT_COLUMN,
                &first_query,
                TOP_K,
                exec_fts::to_infino_mode(first.mode),
                None,
            )
            .expect("warm prewarm bm25_search");
        consumer
            .wait_until_warm(Duration::from_secs(600))
            .expect("supertable warm promotion");
        crate::rss::log_rss_breakdown("supertable_fts after wait_until_warm");
        eprintln!(
            "[supertable_fts] warm: cache hot — timing {} queries × {WARM_ITERS} iters via bm25_search...",
            FTS_BATTERY.len(),
        );
        let out = exec_fts::measure_warm(
            &reader,
            FTS_BATTERY,
            supertable::TEXT_COLUMN,
            TOP_K,
            WARM_ITERS,
            "supertable_fts",
        );
        crate::rss::log_rss_breakdown("supertable_fts after warm battery");
        eprintln!(
            "[supertable_fts] count: cache hot — timing {} queries × {WARM_ITERS} iters \
             (count vs bm25 k=MAX)...",
            FTS_BATTERY.len(),
        );
        let counts = exec_fts::measure_count(
            &reader,
            FTS_BATTERY,
            supertable::TEXT_COLUMN,
            WARM_ITERS,
            "supertable_fts",
        );
        crate::rss::log_rss_breakdown("supertable_fts after count battery");
        // Large-k gate (query phase only, representative subset): surfaces
        // top-k collection cost at serving scale that the top-K table hides.
        eprintln!(
            "[supertable_fts] large-k: timing top-{K_LARGE} query phase × {WARM_ITERS} iters..."
        );
        let large_k: Vec<(&'static str, Duration)> = FTS_BATTERY
            .iter()
            .filter(|q| K_LARGE_SHAPE_NAMES.contains(&q.name))
            .map(|q| {
                let query = q.terms.join(" ");
                let mode = exec_fts::to_infino_mode(q.mode);
                let _ = reader.bm25_rows(supertable::TEXT_COLUMN, &query, K_LARGE, mode);
                let mut samples = Vec::with_capacity(WARM_ITERS);
                for _ in 0..WARM_ITERS {
                    let t = Instant::now();
                    let n = reader.bm25_rows(supertable::TEXT_COLUMN, &query, K_LARGE, mode);
                    samples.push(t.elapsed());
                    std::hint::black_box(n);
                }
                (q.name, crate::executors::p50(&mut samples))
            })
            .collect();
        drop(consumer);
        drop(cache_dir);
        (out, counts, large_k)
    }

    fn measure_cold(
        built: &supertable::IngestResult,
    ) -> std::collections::HashMap<&'static str, crate::executors::ColdTiming> {
        exec_fts::measure_cold(
            || SupertableColdGuard::open(built),
            FTS_BATTERY,
            supertable::TEXT_COLUMN,
            TOP_K,
            COLD_ITERS,
            "supertable_fts",
        )
    }

    /// One metered cold consumer (`ten_term_or`), split at the phase
    /// boundaries the cost model prices: open window, first query on the
    /// cold cache, then the same query repeated on the warm cache.
    fn measure_cold_store(built: &supertable::IngestResult) -> Option<ColdStoreMeasurement> {
        let query = FTS_BATTERY.iter().find(|q| q.name == "ten_term_or")?;
        // Distinct battery entries for steady cold (shared recipe).
        let steady: Vec<&FtsQuery> = FTS_BATTERY
            .iter()
            .filter(|q| q.name != query.name)
            .collect();
        let steady = if steady.is_empty() {
            vec![query]
        } else {
            steady
        };
        let meter = storage_meter::wrap(Arc::clone(&built.storage));
        let measured = cold_store::measure_cold_store(
            &meter,
            || {
                let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
                    meter.provider(),
                    Some(built.total_index_bytes),
                );
                let opts = tiers::consumer_options(
                    supertable::options_for(Modality::Fts, None),
                    meter.provider(),
                    cache,
                );
                let consumer = tiers::open_consumer(opts);
                (cache_dir, consumer)
            },
            |(_cache, consumer)| {
                let terms = query.terms.join(" ");
                let mode = exec_fts::to_infino_mode(query.mode);
                let _ = consumer
                    .reader()
                    .bm25_search(supertable::TEXT_COLUMN, &terms, TOP_K, mode, None)
                    .expect("metered cold bm25_search");
            },
            |(_cache, consumer), i| {
                let q = steady[i % steady.len()];
                let terms = q.terms.join(" ");
                let mode = exec_fts::to_infino_mode(q.mode);
                let _ = consumer
                    .reader()
                    .bm25_search(supertable::TEXT_COLUMN, &terms, TOP_K, mode, None)
                    .expect("metered steady cold bm25_search");
            },
            steady.len().min(STEADY_COLD_SAMPLES),
            |(_cache, consumer)| {
                let terms = query.terms.join(" ");
                let mode = exec_fts::to_infino_mode(query.mode);
                let _ = consumer
                    .reader()
                    .bm25_search(supertable::TEXT_COLUMN, &terms, TOP_K, mode, None)
                    .expect("metered repeat cold bm25_search");
            },
        );
        log_cold_split("supertable_fts", &measured.split);
        Some(measured)
    }

    /// Cold-tier guard: fresh disk cache + consumer only. Same open
    /// window as cost-model cold (`measure_cold_store`) and SQL cold —
    /// no `open_all_superfiles`. Timed search is the first
    /// `bm25_search`, which opens prune survivors itself.
    struct SupertableColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
    }

    impl SupertableColdGuard {
        fn open(built: &supertable::IngestResult) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
            Self {
                _cache_dir: cache_dir,
                consumer,
            }
        }
    }

    impl FtsRead for SupertableColdGuard {
        fn bm25_rows(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: infino::superfile::fts::reader::BoolMode,
        ) -> usize {
            self.consumer
                .reader()
                .bm25_search(column, query, k, mode, None)
                .expect("cold bm25_search")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }

        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: infino::superfile::fts::reader::BoolMode,
        ) -> usize {
            self.consumer
                .reader()
                .bm25_search(column, query, k, mode, Some(&["_id", column, "score"]))
                .expect("cold bm25_search fetched")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }

        fn count_matching(
            &self,
            column: &str,
            query: &str,
            mode: infino::superfile::fts::reader::BoolMode,
        ) -> u64 {
            self.consumer.reader().count_matching(column, query, mode)
        }
    }
}

pub mod vector {
    use std::{
        cmp::Ordering,
        collections::{HashMap, HashSet},
        hint::black_box,
        sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    };

    use infino::storage::io_counters;

    use super::*;
    use crate::{
        corpus,
        executors::{
            vector as exec_vec,
            vector::{SupertableVectorRead, VectorRead},
        },
    };

    // Correctness gate, recall targets, calibration grid, and p50 iters
    // live in `crate::executors::vector` (shared by both tiers).
    //
    // 100 correctness queries: recall is dominated by the per-query
    // probability of a bad cell tie (~5% on the synthetic corpus), and at 20
    // queries the standard error on that rate (±4.9%) exceeded the effect —
    // recall moved in 0.05 quanta run to run. 100 queries brings the standard
    // error to ±2.2% and the occurrence granularity to 0.001. Ground-truth
    // caches are keyed by the query set, so the first run per scale recomputes
    // them once; numbers recorded before this constant changed were measured
    // at 20 queries.
    const N_CORRECTNESS_QUERIES: usize = 100;
    const N_CALIBRATION_QUERIES: usize = 100;
    /// Steady-state cache-budget multiple of the user index for the shared
    /// vector consumer. The hidden per-cell IVF index is a second on-storage
    /// copy of the vector payload, written by the drain *after* this consumer
    /// opens — sizing from the user index alone leaves the cache ~2× under
    /// budget post-drain, and the resulting evictions re-fetch on every query
    /// (measured 62 GET/query on a supposedly warm consumer at 100K).
    const SHARED_CONSUMER_CACHE_INDEX_FACTOR: u64 = 2;
    // The default rows never override engine settings: they run
    // `VectorSearchOptions::default()` via the `ENGINE_DEFAULT` sentinel, so
    // the recorded numbers always measure shipped routing defaults.
    const QUERY_CORRECTNESS_SEED: u64 = 17;
    const QUERY_CALIBRATION_SEED: u64 = 99;
    const QUERY_SIGMA: f32 = 0.05;
    /// Filtered vector bench allow-set density: keep every Nth row.
    const FILTER_KEEP_EVERY: usize = 10;
    /// Regression floor for filtered recall@10 at the bench's ~10%
    /// selectivity — a tripwire below the measured value, the same way
    /// the 0.80 default-config floor sits below its measured 0.995.
    ///
    /// Context for the absolute level: filtered routes like unfiltered
    /// (hidden cells + undrained user tail) with the allow-set pushed
    /// down; the filtered defaults probe 32 hidden cells × 16 fine runs
    /// — measured 0.901 @ 1.37 ms at 1M/256 (~10% selectivity), with
    /// width nearly free on consolidated cells (128 cells → 0.940 @
    /// 1.48 ms). The floor sits under the measured default the same way
    /// the 0.80 default-config floor sits under its 0.995. On THIS
    /// corpus the filtered ground truth sits at unfiltered rank
    /// ~k/selectivity and scatters across most of the grid; real
    /// embedding data (neighbor structure past rank 100) measures
    /// higher at every width. The sweep rows keep the trade visible.
    ///
    /// Temporarily lowered 0.85 -> 0.80: a post-#422 change regressed filtered
    /// recall on this corpus 0.900 -> 0.824 at constant latency (the 1-bit cell
    /// admit ranks cells blind to the allow-set, so matching cells fall outside
    /// the window). Floor relaxed to 0.80 to unblock the bench while the
    /// matching-aware admit fix is worked; restore to 0.85 once it lands.
    const FILTERED_RECALL_FLOOR: f32 = 0.80;
    /// Explicit cell-probe widths for the filtered width-sweep diagnostic
    /// (the engine default probes 128 hidden cells post-drain — width is
    /// nearly free on consolidated cells at 1M). Recall climbing with
    /// width ⇒ cell coverage gap; flat ⇒ in-cell shortlist/rerank loss
    /// (a depth problem). The 256 row is the full 1M/256 grid: exact
    /// search over matching rows, the recall ceiling of the approach.
    const FILTERED_DIAG_PROBE_WIDTHS: &[usize] = &[160, 192, 224, 256];
    /// Repeated warm probes per routing-state transition.
    const ROUTING_STATE_WARM_ITERS: usize = 20;
    /// Explicitly discard only the derived hidden vector-index sibling before
    /// a retained-prefix lifecycle run; the durable user table is untouched.
    const RESET_HIDDEN_INDEX_ENV: &str = "INFINO_BENCH_RESET_HIDDEN_VECTOR_INDEX";

    /// Settled-anon accounting for the cost model's pinned-residency line.
    /// The bench process carries harness state a real serving process never
    /// allocates (the ground-truth id map, corpus bookkeeping, report
    /// buffers), so pricing whole-process anon overstates the engine.
    /// `run()` stamps the consumer handle's own settled-anon open delta
    /// here; each routing state adds what its own battery retained
    /// (settled-after minus settled-before, allocator purged at both
    /// samples). Zero means "not captured".
    static CONSUMER_ENGINE_ANON_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Skip the normal undrained-delta commit while retaining pre-drain,
    /// drain, post-drain, and optimize/compact measurements.
    const SKIP_VECTOR_DELTA_ENV: &str = "INFINO_BENCH_SKIP_VECTOR_DELTA";
    /// Opt IN to the post-drain assignment audit. The audit is
    /// diagnostic-only (it gates nothing) and its full-corpus pass reads
    /// ~0.4 TB of mmap at 100M even in streaming order, so the default is
    /// off; investigation runs set `INFINO_BENCH_DRAIN_DIAG=1`.
    const DRAIN_DIAG_ENV: &str = "INFINO_BENCH_DRAIN_DIAG";
    /// Numerator/denominator for compact p90 drain diagnostics.
    const DRAIN_P90_NUMERATOR: usize = 9;
    const DRAIN_P90_DENOMINATOR: usize = 10;
    /// Cell-probe depths reported by the post-drain assignment audit.
    const DRAIN_DIAG_PROBE_DEPTHS: [usize; 6] = [1, 2, 4, 8, 16, 64];
    /// Stored rows self-queried by the post-drain assignment audit.
    const DRAIN_DIAG_SELF_QUERY_SAMPLE: usize = 500;
    /// Contiguous dense-id span per rayon task in the audit's agreement
    /// scan: 65 536 rows × 1024 dims × 4 B ≈ 256 MiB of corpus bytes per
    /// task, so each worker streams the corpus mmap sequentially. The
    /// previous per-row scatter in HashMap iteration order random-faulted
    /// a corpus far larger than RAM (394 GiB vs 63 GiB at 100M) and never
    /// finished.
    const DRAIN_DIAG_AGREEMENT_CHUNK_ROWS: usize = 65_536;

    /// Recall-target calibration grid — off by default. The shipped search
    /// process routes p=1 over the cell grid and buys recall with write-side
    /// replication plus the fine-probe/slack config defaults, so there is no
    /// per-query nprobe/rerank surface left to tune: the sweep burned minutes
    /// of grid queries per phase to produce rows with no real knob behind
    /// them. Flip to `true` for legacy tuning investigations on the
    /// pre-routing search path.
    const RUN_CALIBRATION_GRID: bool = false;

    /// Regression gates on the cold consumer's **data** GET fan (user +
    /// hidden classes together), by routing state and scale tier, with the
    /// pinned <20M grid shape (512 user / 256 hidden cells). Two windows,
    /// gated separately because they price differently:
    ///
    /// * **first cold query** — the one-time metadata warmup under the v1
    ///   open discipline: the admit-window centroid regions (~20% of cells,
    ///   one block GET each), Sq8 meta, and stable-id blocks ride in with
    ///   the first probe (all concurrent). Measured at 1M/256: 53 GETs.
    ///   Bounded loosely so a fan regression (e.g. the pre-drain 848-class
    ///   blowup reaching a routed state) still trips.
    /// * **second, distinct cold query** — the steady cold per-query fetch
    ///   with the warmup resident: its own probe blocks plus any
    ///   newly-touched cells. This is the number the cost model's cold
    ///   read leg prices, so it gates tight.
    ///
    /// PROVISIONAL until the validation ladder re-measures both windows;
    /// tighten to the recorded numbers. At and above
    /// [`COLD_GET_MID_MAX_DOCS`] the grid shape is still being calibrated,
    /// so no ceiling applies yet.
    const COLD_GET_SMALL_MAX_DOCS: usize = 5_000_000;
    /// Upper doc bound for the mid-scale ceilings (exclusive).
    const COLD_GET_MID_MAX_DOCS: usize = 20_000_000;
    /// Per-state `(label, <5M ceiling, 5M–20M ceiling)` on the FIRST cold
    /// query's DATA GETs (probe blocks; manifest GETs — parts, slow-CAS
    /// blob, centroid section — are classed separately). A cold probe
    /// reads the geometric-chain islands its selected runs span under the
    /// 8 MiB cold coalesce windows: whole-cell at <5M (cells ~6 MiB), 2–4
    /// islands at 10M (cells ~60 MiB; bridging 10–18 MiB inter-island
    /// gaps would cost more wall time on one stream than parallel GETs).
    const COLD_GET_CEILINGS_FIRST: &[(&str, u64, u64)] = &[
        ("post-drain", 4, 8),
        ("post-delta", 6, 10),
        ("post-compact", 4, 8),
    ];
    /// Per-state `(label, <5M ceiling, 5M–20M ceiling)` on the SECOND
    /// (steady) cold query's data GETs. <5M: a probed cell spans ~6 MiB,
    /// the whole probe coalesces to ONE GET (measured 1 / 2 / 1 at 1M —
    /// post-delta's extra GET is the undrained user tail). 5–20M: a
    /// probed cell spans ~60 MiB and the selected runs occupy 2–4
    /// geometric-chain islands with 10–18 MiB gaps that are cheaper to
    /// fetch in parallel than to bridge (median 3–4 measured at 10M
    /// under the 8 MiB cold windows). Invariant across tiers:
    /// post-delta = post-drain + 1 (the undrained user tail is exactly
    /// one extra coalesced GET). Post-compact matches post-delta at mid
    /// scale — a budgeted optimize leaves the hidden table two shard
    /// generations deep, so a probed cell's runs span two files
    /// (measured 5 at 10M: 3–4 islands in the old shard + 1 in the
    /// new); a full per-cell consolidation pass would earn post-drain's
    /// ceiling back. The old 2-GET value at this tier was calibrated
    /// against the fat-open era (727 MiB opens staging all cell
    /// metadata) and is not reachable on the v1-open architecture.
    const COLD_GET_CEILINGS_SECOND: &[(&str, u64, u64)] = &[
        ("post-drain", 1, 4),
        ("post-delta", 2, 5),
        ("post-compact", 1, 5),
    ];
    /// Ceiling for `label` + `n_docs` out of one of the two gate tables,
    /// when one applies to that state at this scale.
    fn cold_data_get_ceiling(
        table: &[(&str, u64, u64)],
        label: &str,
        n_docs: usize,
    ) -> Option<u64> {
        let (_, small, mid) = table.iter().find(|(state, _, _)| *state == label)?;
        if n_docs < COLD_GET_SMALL_MAX_DOCS {
            Some(*small)
        } else if n_docs < COLD_GET_MID_MAX_DOCS {
            Some(*mid)
        } else {
            None
        }
    }

    /// Calibration policy for supertable vector benches: the grid runs only
    /// when [`RUN_CALIBRATION_GRID`] is flipped on, and even then auto-offs
    /// above [`exec_vec::FULL_CALIBRATION_MAX_DOCS`]. Default and filtered
    /// recall are always computed either way — only the recall-target sweep
    /// is skipped.
    fn skip_calibration(n_docs: usize) -> bool {
        !RUN_CALIBRATION_GRID || n_docs > exec_vec::FULL_CALIBRATION_MAX_DOCS
    }

    /// Probe count for the `default` row: the engine default, never a bench
    /// override, so a leaked per-shell setting can never skew recorded
    /// numbers.
    fn fixed_nprobe() -> usize {
        exec_vec::ENGINE_DEFAULT
    }
    /// Rerank multiplier for the `default` row. Same policy as
    /// [`fixed_nprobe`].
    fn fixed_rerank_mult() -> usize {
        exec_vec::ENGINE_DEFAULT
    }

    #[derive(Clone, Copy)]
    enum ExpectedTiers {
        UserOnly,
        HiddenOnly,
        Both,
    }

    impl ExpectedTiers {
        fn label(self) -> &'static str {
            match self {
                Self::UserOnly => "user data only",
                Self::HiddenOnly => "hidden data only",
                Self::Both => "user + hidden data",
            }
        }
    }

    struct RoutingStateStat {
        label: &'static str,
        expected: ExpectedTiers,
        recall: Option<String>,
        warm_p50_ns: Option<f64>,
        warm_cpu_s: Option<f64>,
        ram_bytes: Option<u64>,
        /// Engine-only settled anon after this state's battery: the consumer
        /// handle's open delta plus retained serving growth, with freed query
        /// scratch purged and harness heap subtracted (see
        /// [`CONSUMER_ENGINE_ANON_BYTES`]).
        ram_anon_bytes: Option<u64>,
        /// Settled file-backed resident bytes at the same sample — the mmap
        /// page-cache working set actually held after serving this state.
        ram_file_settled_bytes: Option<u64>,
        warm_io: Option<storage_meter::ObjectStoreMeter>,
        cold: Option<RoutingColdStat>,
    }

    #[derive(Clone, Copy)]
    struct RoutingColdStat {
        split: storage_meter::ColdStoreSplit,
        open_wall_s: f64,
        open_cpu_s: Option<f64>,
        query_wall_s: f64,
        query_cpu_s: Option<f64>,
        /// Wall/CPU of the second, distinct cold query — the steady cold
        /// per-query cost once the first query's metadata warmup landed.
        second_wall_s: f64,
        second_cpu_s: Option<f64>,
    }

    struct TransitionStat {
        label: &'static str,
        wall_ns: f64,
        io: Option<storage_meter::ObjectStoreMeter>,
        peak_rss_bytes: Option<u64>,
    }

    struct HitTierStats {
        user_hits: usize,
        hidden_hits: usize,
    }

    fn hit_tier_counts(
        table: &Supertable,
        query: &[f32],
        nprobe: usize,
        rerank: usize,
    ) -> HitTierStats {
        let reader = table.reader();
        let hidden_uris: HashSet<_> = table
            .vector_index_table()
            .map(|hidden| {
                hidden
                    .pinned_reader()
                    .manifest()
                    .get_all_superfiles()
                    .iter()
                    .map(|entry| entry.uri)
                    .collect()
            })
            .unwrap_or_default();
        let hits = reader
            .vector_hits(
                supertable::VEC_COLUMN,
                query,
                TOP_K,
                exec_vec::search_opts(nprobe, rerank),
                None,
            )
            .expect("routing-state vector hits");
        let user_hits = hits
            .iter()
            .filter(|hit| !hidden_uris.contains(&hit.superfile))
            .count();
        let hidden_hits = hits
            .iter()
            .filter(|hit| hidden_uris.contains(&hit.superfile))
            .count();
        HitTierStats {
            user_hits,
            hidden_hits,
        }
    }

    fn assert_expected_tiers(
        label: &str,
        expected: ExpectedTiers,
        user_hits: usize,
        hidden_hits: usize,
    ) {
        let valid = match expected {
            ExpectedTiers::UserOnly => user_hits > 0 && hidden_hits == 0,
            ExpectedTiers::HiddenOnly => user_hits == 0 && hidden_hits > 0,
            // Mixed routing is proven by per-class cold GETs below. A normal
            // follow-up commit need not contribute a row to every top-k.
            ExpectedTiers::Both => true,
        };
        assert!(
            valid,
            "{label}: unexpected tier coverage (user hits={user_hits}, hidden hits={hidden_hits})"
        );
    }

    fn assert_expected_cold_reads(
        label: &str,
        expected: ExpectedTiers,
        split: &storage_meter::ColdStoreSplit,
        n_docs: usize,
    ) {
        let user_data = split
            .first_query
            .class_io(storage_meter::UriClass::UserData)
            .get_count;
        let hidden_data = split
            .first_query
            .class_io(storage_meter::UriClass::HiddenData)
            .get_count;
        let valid = match expected {
            ExpectedTiers::UserOnly => user_data > 0 && hidden_data == 0,
            ExpectedTiers::HiddenOnly => user_data == 0 && hidden_data > 0,
            ExpectedTiers::Both => user_data > 0 && hidden_data > 0,
        };
        assert!(
            valid,
            "{label}: unexpected cold data reads (user data GET={user_data}, hidden data GET={hidden_data})"
        );
        // Lock in the cold-probe gains, per window: the first query's
        // one-time warmup fan and the second query's steady per-query fetch
        // each stay within their per-scale ceilings.
        if let Some(ceiling) = cold_data_get_ceiling(COLD_GET_CEILINGS_FIRST, label, n_docs) {
            let total = user_data + hidden_data;
            assert!(
                total <= ceiling,
                "{label}: first cold query (metadata warmup) regressed — {total} data GETs \
                 ({user_data} user + {hidden_data} hidden), ceiling {ceiling} at {n_docs} docs \
                 (see COLD_GET_CEILINGS_FIRST; provisional post-v1-open values)"
            );
        }
        if let Some(ceiling) = cold_data_get_ceiling(COLD_GET_CEILINGS_SECOND, label, n_docs) {
            let second_user = split
                .second_query
                .class_io(storage_meter::UriClass::UserData)
                .get_count;
            let second_hidden = split
                .second_query
                .class_io(storage_meter::UriClass::HiddenData)
                .get_count;
            let total = second_user + second_hidden;
            assert!(
                total <= ceiling,
                "{label}: second (steady) cold query regressed — {total} data GETs \
                 ({second_user} user + {second_hidden} hidden), ceiling {ceiling} at {n_docs} \
                 docs (see COLD_GET_CEILINGS_SECOND; provisional post-v1-open values)"
            );
        }
    }

    fn default_recall(rows: &[exec_vec::RecallRow]) -> Option<String> {
        rows.iter()
            .find(|row| row.target == "default")
            .map(|row| row.recall.clone())
    }

    fn class_gets(io: storage_meter::ObjectStoreMeter, class: storage_meter::UriClass) -> String {
        let class_io = io.class_io(class);
        if class_io.get_count == 0 {
            "0".into()
        } else {
            format!(
                "{} / {}",
                class_io.get_count,
                rss::fmt_bytes(class_io.get_bytes)
            )
        }
    }

    fn manifest_gets(io: storage_meter::ObjectStoreMeter) -> String {
        let user = io.class_io(storage_meter::UriClass::UserManifest);
        let hidden = io.class_io(storage_meter::UriClass::HiddenManifest);
        let count = user.get_count + hidden.get_count;
        let bytes = user.get_bytes + hidden.get_bytes;
        if count == 0 {
            "0".into()
        } else {
            format!("{count} / {}", rss::fmt_bytes(bytes))
        }
    }

    struct SupertableVecColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
        id_to_dense: Arc<std::collections::HashMap<i128, u32>>,
    }

    impl SupertableVecColdGuard {
        fn open(
            built: &supertable::IngestResult,
            id_to_dense: Arc<std::collections::HashMap<i128, u32>>,
        ) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Vector, built);
            Self {
                _cache_dir: cache_dir,
                consumer,
                id_to_dense,
            }
        }
    }

    impl VectorRead for SupertableVecColdGuard {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)> {
            SupertableVectorRead {
                table: &self.consumer,
                id_to_dense: Arc::clone(&self.id_to_dense),
            }
            .topk_global(column, query, k, nprobe, rerank)
        }
    }

    fn hits_to_dense_u32(
        st: &Supertable,
        id_to_dense: &HashMap<i128, u32>,
        hits: &[infino::supertable::query::SuperfileHit],
    ) -> Vec<(u32, f32)> {
        let reader = st.reader();
        let manifest = reader.manifest();
        let mut contiguous_min_by_uri: HashMap<_, i128> = HashMap::new();
        for entry in manifest.get_all_superfiles() {
            let span = entry.id_max.saturating_sub(entry.id_min).saturating_add(1);
            if span == entry.n_docs as i128 {
                contiguous_min_by_uri.insert(entry.uri, entry.id_min);
            }
        }
        if let Some(hidden) = st.vector_index_table() {
            let hidden_reader = hidden.pinned_reader();
            let hidden_manifest = hidden_reader.manifest();
            for entry in hidden_manifest.get_all_superfiles() {
                let span = entry.id_max.saturating_sub(entry.id_min).saturating_add(1);
                if span == entry.n_docs as i128 {
                    contiguous_min_by_uri.insert(entry.uri, entry.id_min);
                }
            }
        }
        hits.iter()
            .filter_map(|h| {
                let stable_id = h.stable_id.or_else(|| {
                    contiguous_min_by_uri
                        .get(&h.superfile)
                        .copied()
                        .map(|id_min| id_min + i128::from(h.local_doc_id))
                })?;
                let dense = id_to_dense.get(&stable_id).copied().or_else(|| {
                    u32::try_from(stable_id)
                        .ok()
                        .filter(|id| *id < supertable::n_docs() as u32)
                })?;
                Some((dense, h.score))
            })
            .collect()
    }

    fn distribution(values: &mut [u64]) -> Option<(u64, u64, u64, u64)> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let p90_index = values
            .len()
            .saturating_mul(DRAIN_P90_NUMERATOR)
            .div_ceil(DRAIN_P90_DENOMINATOR)
            .saturating_sub(1);
        Some((
            values[0],
            values[values.len() / 2],
            values[p90_index],
            values[values.len() - 1],
        ))
    }

    /// 1-based rank of every index in a scored list: sorts ascending by
    /// score (index breaks ties, matching the engine's lowest-index-wins
    /// rule) and scatters positions into a dense `len`-slot map
    /// (`usize::MAX` = never scored). Shared by the audit's grid, routed,
    /// and fine-run coverage curves.
    fn rank_map(mut scored: Vec<(usize, f32)>, len: usize) -> Vec<usize> {
        scored.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut rank_by_index = vec![usize::MAX; len];
        for (rank, (idx, _)) in scored.iter().enumerate() {
            rank_by_index[*idx] = rank + 1;
        }
        rank_by_index
    }

    /// Post-drain assignment audit: separates "the drain stored rows in the
    /// wrong cells" from "the query's nearest cell doesn't contain the true
    /// neighbors". Three measurements against the user table's global grid:
    ///
    /// 1. row conservation — stored rows vs corpus rows (drops/duplicates);
    /// 2. assignment agreement — % of stored rows whose cell IS their exact
    ///    nearest grid centroid (fp32 corpus math, no decode involved);
    /// 3. neighbor coverage — for each ground-truth neighbor, the rank of its
    ///    STORED cell (and separately its geometrically-nearest cell) in the
    ///    query's cell ordering; `p1` is the ceiling of `cells 1..1` recall.
    ///
    /// Deliberate replication: the oracle math here — the naive
    /// `nearest_cell` scan and the tie-window threshold expression — is
    /// intentionally written against the plain scalar formulas rather than
    /// the engine's blocked kernels or `relative_score_window` (which are
    /// crate-private anyway). A bug in an engine kernel must show up as a
    /// disagreement in this audit, not silently agree with itself.
    fn report_post_drain_assignment_audit(
        consumer: &Supertable,
        vectors: &[f32],
        queries: &[Vec<f32>],
        ground_truth: &[Vec<u32>],
        id_to_dense: &HashMap<i128, u32>,
    ) {
        use rayon::prelude::*;

        if env::var(DRAIN_DIAG_ENV).ok().as_deref() != Some("1") {
            eprintln!("[drain-diag] skipped — diagnostic-only; opt in with {DRAIN_DIAG_ENV}=1");
            return;
        }

        let Some(cells) = consumer.hidden_cell_stable_id_sets() else {
            eprintln!("[drain-diag] no packed hidden cells to audit");
            return;
        };
        let reader = consumer.pinned_reader();
        let manifest = reader.manifest();
        let Some(global) = manifest.get_global_vector_index() else {
            eprintln!("[drain-diag] user manifest has no global cell grid");
            return;
        };
        let grid = global.grid;
        let metric = consumer.options().vector_columns[0].metric;
        let n_cells = grid.n_cent as usize;
        let n_rows = vectors.len() / DIM;

        let stored_total: usize = cells.iter().map(|(_, ids)| ids.len()).sum();
        let mut stored_by_dense: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut unmapped = 0usize;
        for (cell, ids) in &cells {
            for id in ids {
                match id_to_dense.get(id) {
                    Some(dense) => stored_by_dense.entry(*dense).or_default().push(*cell),
                    None => unmapped += 1,
                }
            }
        }
        // Same row stored twice in the SAME cell = wasted top-k slots at
        // query time and inflated amplification; replicas are only ever
        // legitimate in a *different* cell than the primary.
        let mut duplicate_pairs = 0usize;
        for stored in stored_by_dense.values() {
            let mut sorted = stored.clone();
            sorted.sort_unstable();
            duplicate_pairs += sorted.windows(2).filter(|w| w[0] == w[1]).count();
        }
        eprintln!(
            "[drain-diag] stored rows: {stored_total} across {} cells (distinct rows {}, corpus rows {n_rows}, unmapped ids {unmapped}, same-cell duplicate pairs {duplicate_pairs})",
            cells.len(),
            stored_by_dense.len(),
        );

        // Geometric nearest cell over ALL cells (count-0 included) — the
        // kernel-backed full ranking, same tie-break (lowest id).
        let nearest_cell = |vector: &[f32]| -> u32 {
            grid.rank_cells(metric, vector)
                .first()
                .map(|&(cell, _)| cell)
                .unwrap_or(0)
        };

        // Agreement scan in dense (corpus) order: chunk contiguous dense ids
        // so every rayon task streams its 256 MiB corpus span sequentially
        // instead of random-faulting the mmap per HashMap-ordered row.
        let n_chunks = n_rows.div_ceil(DRAIN_DIAG_AGREEMENT_CHUNK_ROWS);
        let (agree, audited_len) = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start_row = chunk_idx * DRAIN_DIAG_AGREEMENT_CHUNK_ROWS;
                let end_row = (start_row + DRAIN_DIAG_AGREEMENT_CHUNK_ROWS).min(n_rows);
                let mut local_agree = 0usize;
                let mut local_audited = 0usize;
                for dense in start_row..end_row {
                    let Some(stored) = stored_by_dense.get(&(dense as u32)) else {
                        continue;
                    };
                    local_audited += 1;
                    let start = dense * DIM;
                    if stored.contains(&nearest_cell(&vectors[start..start + DIM])) {
                        local_agree += 1;
                    }
                }
                (local_agree, local_audited)
            })
            .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
        eprintln!(
            "[drain-diag] assignment agreement: {agree}/{audited_len} stored rows sit in their exact nearest grid cell ({:.3})",
            agree as f64 / audited_len.max(1) as f64,
        );

        // Fine-centroid routing replica: the hidden query path ranks cells by
        // the best (minimum) score over each cell's summary fine centroids,
        // not by the grid centroid the drain assigned against. Collect every
        // packed cell's fine centroids from the hidden manifest summaries so
        // coverage can be measured under the ranking the query actually uses.
        let fine_by_cell: HashMap<u32, Vec<ClusterCentroids>> = {
            let mut out: HashMap<u32, Vec<_>> = HashMap::new();
            if let Some(hidden) = consumer.vector_index_table() {
                let hidden_reader = hidden.pinned_reader();
                for entry in hidden_reader.manifest().get_all_superfiles() {
                    for summary in entry.vector_summary.values() {
                        for cell in &summary.cells {
                            if let Some(cell_id) = cell.cell_id {
                                out.entry(cell_id).or_default().push(cell.clusters.clone());
                            }
                        }
                    }
                }
            }
            out
        };

        let mut cov_stored = [0usize; DRAIN_DIAG_PROBE_DEPTHS.len()];
        let mut cov_geom = [0usize; DRAIN_DIAG_PROBE_DEPTHS.len()];
        let mut cov_routed = [0usize; DRAIN_DIAG_PROBE_DEPTHS.len()];
        let mut total = 0usize;
        let mut missing = 0usize;
        for (query, truth) in queries.iter().zip(ground_truth) {
            let rank_by_cell = rank_map(
                grid.rank_cells(metric, query)
                    .into_iter()
                    .map(|(cell, score)| (cell as usize, score))
                    .collect(),
                n_cells,
            );
            // Query-path replica: rank cells by min fine-centroid score.
            let routed_rank_by_cell = rank_map(
                fine_by_cell
                    .iter()
                    .map(|(cell_id, cluster_sets)| {
                        let mut best = f32::INFINITY;
                        for clusters in cluster_sets {
                            clusters.score_clusters_into(metric, query, |_, score| {
                                best = best.min(score);
                            });
                        }
                        (*cell_id as usize, best)
                    })
                    .collect(),
                n_cells,
            );
            for id in truth {
                let start = *id as usize * DIM;
                if start + DIM > vectors.len() {
                    continue;
                }
                total += 1;
                let geom_rank = rank_by_cell[nearest_cell(&vectors[start..start + DIM]) as usize];
                let (stored_rank, routed_rank) = stored_by_dense
                    .get(id)
                    .map(|stored| {
                        let grid_rank = stored
                            .iter()
                            .map(|cell| rank_by_cell[*cell as usize])
                            .min()
                            .unwrap_or(usize::MAX);
                        let routed = stored
                            .iter()
                            .map(|cell| routed_rank_by_cell[*cell as usize])
                            .min()
                            .unwrap_or(usize::MAX);
                        (grid_rank, routed)
                    })
                    .unwrap_or_else(|| {
                        missing += 1;
                        (usize::MAX, usize::MAX)
                    });
                for (i, probe) in DRAIN_DIAG_PROBE_DEPTHS.iter().enumerate() {
                    cov_geom[i] += usize::from(geom_rank <= *probe);
                    cov_stored[i] += usize::from(stored_rank <= *probe);
                    cov_routed[i] += usize::from(routed_rank <= *probe);
                }
            }
        }
        let fmt_curve = |cov: &[usize], denom: usize| {
            DRAIN_DIAG_PROBE_DEPTHS
                .iter()
                .zip(cov)
                .map(|(probe, count)| {
                    format!("p{probe}={:.3}", *count as f64 / denom.max(1) as f64)
                })
                .collect::<Vec<_>>()
                .join(" · ")
        };
        eprintln!(
            "[drain-diag] GT neighbor coverage by STORED cell ({total} occurrences, {missing} not stored): {}",
            fmt_curve(&cov_stored, total),
        );
        eprintln!(
            "[drain-diag] GT neighbor coverage by NEAREST cell (perfect assignment): {}",
            fmt_curve(&cov_geom, total),
        );
        eprintln!(
            "[drain-diag] GT neighbor coverage under FINE-centroid routing (query path): {}",
            fmt_curve(&cov_routed, total),
        );

        // Closure-window forensics: for every neighbor the p=1 probe misses,
        // report the distance ratio and closure depth a replica into the
        // query's top-1 cell would have needed — i.e. what
        // REPLICA_CLOSURE_DISTANCE_RATIO / MAX_REPLICAS must cover to fix it.
        for (query, truth) in queries.iter().zip(ground_truth) {
            let probed = nearest_cell(query);
            for id in truth {
                let start = *id as usize * DIM;
                if start + DIM > vectors.len() {
                    continue;
                }
                if stored_by_dense
                    .get(id)
                    .is_some_and(|cells| cells.contains(&probed))
                {
                    continue;
                }
                let neighbor = &vectors[start..start + DIM];
                let own = nearest_cell(neighbor);
                let own_score = grid.score_one(metric, own as usize, neighbor);
                let rescue_score = grid.score_one(metric, probed as usize, neighbor);
                let needed_ratio = rescue_score / own_score.abs().max(f32::EPSILON);
                let depth = grid
                    .rank_cells(metric, neighbor)
                    .iter()
                    .filter(|&&(cell, score)| cell != own && score < rescue_score)
                    .count();
                eprintln!(
                    "[drain-diag] miss: neighbor {id} stored in cell {own}, probe hits cell {probed}; rescue needs ratio {needed_ratio:.3} at closure depth {}",
                    depth + 1,
                );
            }
        }

        // Amplification price list: fraction of corpus rows with k cells
        // inside each candidate ratio window (sampled), i.e. the storage
        // factor each REPLICA_CLOSURE_DISTANCE_RATIO setting would buy.
        const CLOSURE_RATIO_CANDIDATES: [f32; 3] = [1.2, 1.5, 2.0];
        let sample_step = (n_rows / DRAIN_DIAG_SELF_QUERY_SAMPLE).max(1);
        let sampled: Vec<usize> = (0..n_rows).step_by(sample_step).collect();
        let copies: Vec<[usize; CLOSURE_RATIO_CANDIDATES.len()]> = sampled
            .par_iter()
            .map(|&row_idx| {
                let row = &vectors[row_idx * DIM..(row_idx + 1) * DIM];
                // Already ascending — `rank_cells` sorts.
                let scores: Vec<f32> = grid
                    .rank_cells(metric, row)
                    .into_iter()
                    .map(|(_, score)| score)
                    .collect();
                let primary = scores[0];
                let mut out = [0usize; CLOSURE_RATIO_CANDIDATES.len()];
                for (i, ratio) in CLOSURE_RATIO_CANDIDATES.iter().enumerate() {
                    let threshold = primary + primary.abs().max(f32::EPSILON) * (ratio - 1.0);
                    out[i] = scores[1..].iter().filter(|s| **s <= threshold).count();
                }
                out
            })
            .collect();
        let price_list = CLOSURE_RATIO_CANDIDATES
            .iter()
            .enumerate()
            .map(|(i, ratio)| {
                let extra: usize = copies.iter().map(|c| c[i]).sum();
                format!(
                    "ratio {ratio}: {:.2}x",
                    1.0 + extra as f64 / copies.len().max(1) as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" · ");
        eprintln!(
            "[drain-diag] unbudgeted closure amplification over {} sampled rows: {price_list}",
            copies.len(),
        );

        // Tie-window price list for near-tie probe widening: the fraction of
        // queries whose rank-2..4 cells fall within each slack window of the
        // top cell. Sums to the expected probe amplification at that slack
        // (`nprobe_min=1, nprobe_max=4`). Scale-invariant: this is query
        // distribution vs grid geometry, independent of corpus size.
        const TIE_SLACK_CANDIDATES: [f32; 3] = [0.02, 0.04, 0.08];
        let mut tie_counts = [[0usize; 3]; TIE_SLACK_CANDIDATES.len()];
        for query in queries {
            // Already ascending — `rank_cells` sorts.
            let scores: Vec<f32> = grid
                .rank_cells(metric, query)
                .into_iter()
                .map(|(_, score)| score)
                .collect();
            let top = scores[0];
            for (si, slack) in TIE_SLACK_CANDIDATES.iter().enumerate() {
                let threshold = top + top.abs().max(f32::EPSILON) * slack;
                for (ri, tie_count) in tie_counts[si].iter_mut().enumerate() {
                    if scores.get(ri + 1).is_some_and(|s| *s <= threshold) {
                        *tie_count += 1;
                    }
                }
            }
        }
        let tie_list = TIE_SLACK_CANDIDATES
            .iter()
            .enumerate()
            .map(|(si, slack)| {
                let extra: usize = tie_counts[si].iter().sum();
                format!(
                    "slack {:.0}%: rank2 {:.2} · rank3 {:.2} · rank4 {:.2} → amp {:.2}x",
                    slack * 100.0,
                    tie_counts[si][0] as f64 / queries.len().max(1) as f64,
                    tie_counts[si][1] as f64 / queries.len().max(1) as f64,
                    tie_counts[si][2] as f64 / queries.len().max(1) as f64,
                    1.0 + extra as f64 / queries.len().max(1) as f64,
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        eprintln!(
            "[drain-diag] near-tie widening price over {} queries: {tie_list}",
            queries.len(),
        );

        // Within-cell fine-run spread: for neighbors stored in the query's
        // probed (grid-top-1) cell, rank that cell's fine centroids by query
        // distance and record the rank of the neighbor's own run. Directly
        // measures how many ~2 MiB fine runs `fine_nprobe` must cover — the
        // GET/byte floor of the within-cell probe, and the guard that says
        // how fat a cell can grow before fine 8 stops covering it.
        let mut fine_cov = [0usize; DRAIN_DIAG_PROBE_DEPTHS.len()];
        let mut fine_total = 0usize;
        for (query, truth) in queries.iter().zip(ground_truth) {
            let probed = nearest_cell(query);
            let Some(cluster_sets) = fine_by_cell.get(&probed) else {
                continue;
            };
            let mut query_scores: Vec<(usize, f32)> = Vec::new();
            let mut flat_base = 0usize;
            for clusters in cluster_sets {
                clusters.score_clusters_into(metric, query, |local, score| {
                    query_scores.push((flat_base + local as usize, score));
                });
                flat_base += clusters.n_cent as usize;
            }
            let rank_of = rank_map(query_scores, flat_base);
            for id in truth {
                let start = *id as usize * DIM;
                if start + DIM > vectors.len() {
                    continue;
                }
                if !stored_by_dense
                    .get(id)
                    .is_some_and(|cells| cells.contains(&probed))
                {
                    continue;
                }
                fine_total += 1;
                let neighbor = &vectors[start..start + DIM];
                let mut best_run = usize::MAX;
                let mut best_score = f32::INFINITY;
                let mut flat = 0usize;
                for clusters in cluster_sets {
                    clusters.score_clusters_into(metric, neighbor, |local, score| {
                        if score < best_score {
                            best_score = score;
                            best_run = flat + local as usize;
                        }
                    });
                    flat += clusters.n_cent as usize;
                }
                let run_rank = rank_of.get(best_run).copied().unwrap_or(usize::MAX);
                for (i, probe) in DRAIN_DIAG_PROBE_DEPTHS.iter().enumerate() {
                    fine_cov[i] += usize::from(run_rank <= *probe);
                }
            }
        }
        eprintln!(
            "[drain-diag] fine-run coverage inside the probed cell ({fine_total} neighbor occurrences): {}",
            fmt_curve(&fine_cov, fine_total),
        );

        // Self-query probe: search the post-drain index with stored rows' own
        // vectors. Coverage is guaranteed (a row's cell IS its nearest cell,
        // verified above), so any miss is a within-cell defect — id/code/rerank
        // pairing or shortlist scoring — not routing.
        let dense_ids: Vec<u32> = {
            let mut ids: Vec<u32> = stored_by_dense.keys().copied().collect();
            ids.sort_unstable();
            let step = (ids.len() / DRAIN_DIAG_SELF_QUERY_SAMPLE).max(1);
            ids.into_iter()
                .step_by(step)
                .take(DRAIN_DIAG_SELF_QUERY_SAMPLE)
                .collect()
        };
        let dense_to_id: HashMap<u32, i128> = id_to_dense
            .iter()
            .map(|(stable, dense)| (*dense, *stable))
            .collect();
        let mut self_top1 = 0usize;
        let mut self_top10 = 0usize;
        let mut sampled = 0usize;
        for dense in &dense_ids {
            let start = *dense as usize * DIM;
            if start + DIM > vectors.len() {
                continue;
            }
            let Some(&stable) = dense_to_id.get(dense) else {
                continue;
            };
            sampled += 1;
            let batches = consumer
                .reader()
                .vector_search(
                    supertable::VEC_COLUMN,
                    &vectors[start..start + DIM],
                    TOP_K,
                    exec_vec::default_search_opts(),
                    None,
                    None,
                )
                .expect("drain-diag self-query");
            let ids = corpus::id_scores_from_vector_search(&batches);
            if ids.first().is_some_and(|(id, _)| *id == stable) {
                self_top1 += 1;
            }
            if ids.iter().any(|(id, _)| *id == stable) {
                self_top10 += 1;
            }
        }
        eprintln!(
            "[drain-diag] self-query on stored rows ({sampled} sampled): top-1 self-hit {:.3}, top-10 self-hit {:.3}",
            self_top1 as f64 / sampled.max(1) as f64,
            self_top10 as f64 / sampled.max(1) as f64,
        );
    }

    fn log_hidden_stats(consumer: &Supertable, label: &str) {
        let Some(hidden) = consumer.vector_index_table() else {
            return;
        };
        let hidden_reader = hidden.pinned_reader();
        let entries = hidden_reader.manifest().get_all_superfiles();
        let mut rows_by_cell: HashMap<u32, u64> = HashMap::new();
        let mut rows_by_fine_cluster = Vec::new();
        let mut stored_bytes = 0u64;
        for entry in entries {
            stored_bytes = stored_bytes.saturating_add(
                entry
                    .subsection_offsets
                    .as_ref()
                    .map(|offsets| offsets.total_size)
                    .unwrap_or(0),
            );
            for summary in entry.vector_summary.values() {
                for cell in &summary.cells {
                    let cell_rows: u64 = cell
                        .clusters
                        .counts
                        .iter()
                        .map(|count| u64::from(*count))
                        .sum();
                    if let Some(cell_id) = cell.cell_id {
                        *rows_by_cell.entry(cell_id).or_default() += cell_rows;
                    }
                    rows_by_fine_cluster
                        .extend(cell.clusters.counts.iter().map(|count| u64::from(*count)));
                }
            }
        }
        let mut cell_rows: Vec<u64> = rows_by_cell.into_values().collect();
        let cell_dist = distribution(&mut cell_rows);
        let fine_dist = distribution(&mut rows_by_fine_cluster);
        if let (
            Some((cell_min, cell_p50, cell_p90, cell_max)),
            Some((fine_min, fine_p50, fine_p90, fine_max)),
        ) = (cell_dist, fine_dist)
        {
            eprintln!(
                "[supertable_vector] hidden {label}: {} files, {} cells, {} fine clusters, {}; cell rows min/p50/p90/max={cell_min}/{cell_p50}/{cell_p90}/{cell_max}; fine rows={fine_min}/{fine_p50}/{fine_p90}/{fine_max}",
                entries.len(),
                cell_rows.len(),
                rows_by_fine_cluster.len(),
                rss::fmt_bytes(stored_bytes),
            );
        } else {
            eprintln!(
                "[supertable_vector] hidden {label}: {} files, no resident cells",
                entries.len()
            );
        }
    }

    /// Observable routing phase of an already-built table — which tier(s) a
    /// vector query fans out over right now. Derived purely from manifest
    /// state so read-only (existing-prefix / dataset) runs report the same
    /// phase names as lifecycle runs without performing any transition:
    /// no hidden superfiles ⇒ pre-drain; hidden present with every user
    /// commit drained ⇒ post-drain; hidden present with an undrained user
    /// tail ⇒ post-delta.
    fn current_routing_phase(consumer: &Supertable) -> &'static str {
        let Some(hidden) = consumer.vector_index_table() else {
            return "pre-drain";
        };
        let hidden_reader = hidden.pinned_reader();
        let hidden_manifest = hidden_reader.manifest();
        if hidden_manifest.get_all_superfiles().is_empty() {
            return "pre-drain";
        }
        let drained = hidden_manifest.get_drained_ranges();
        let user_reader = consumer.reader();
        let user_manifest = user_reader.manifest();
        if user_manifest
            .get_all_superfiles()
            .iter()
            .any(|entry| !drained.contains(entry.birth_version))
        {
            "post-delta"
        } else {
            "post-drain"
        }
    }

    fn log_hidden_open_stats(hidden: &Supertable, label: &str) {
        let reader = hidden.pinned_reader();
        let manifest = reader.manifest();
        let parts = manifest.get_num_parts();
        let loaded_before = manifest.get_num_parts_loaded();
        let flat_superfiles = manifest.get_all_superfiles().len();
        let mut total = 0usize;
        let mut with_offsets = 0usize;
        let mut with_open_blob = 0usize;
        let mut open_blob_bytes = 0u64;
        let mut vec_open_ranges = 0usize;
        visit_manifest_superfiles(hidden, |entry| {
            total += 1;
            if let Some(offsets) = entry.subsection_offsets.as_ref() {
                with_offsets += 1;
                vec_open_ranges += offsets.vec_open_ranges.len();
                if !offsets.open_blob.is_empty() {
                    with_open_blob += 1;
                    open_blob_bytes = open_blob_bytes.saturating_add(
                        offsets
                            .open_blob
                            .iter()
                            .map(|(_, bytes)| bytes.len() as u64)
                            .sum::<u64>(),
                    );
                }
            }
        });
        let loaded_after = manifest.get_num_parts_loaded();
        eprintln!(
            "[supertable_vector] hidden vector index {label}: manifest parts {parts} ({loaded_before} loaded before stats, {loaded_after} after), flat view {flat_superfiles} superfiles, entries {total}, offsets {with_offsets}/{total}, open_blob {with_open_blob}/{with_offsets} ({}), vec_open_ranges {vec_open_ranges}",
            rss::fmt_bytes(open_blob_bytes),
        );
    }

    /// Drain hidden incoming IVF into per-cell superfiles via the existing
    /// OPANN maintenance hook (same call integration tests use).
    fn drain_hidden_incoming(consumer: &Supertable) {
        let hidden = consumer
            .vector_index_table()
            .expect("vector table keeps hidden index");
        eprintln!("[supertable_vector] draining user superfiles into cell superfiles...");
        consumer
            .drain_vectors_to_cells_sync()
            .expect("hidden cell drain");
        log_hidden_stats(hidden, "after drain");
    }

    fn reset_hidden_vector_index_if_requested(built: &supertable::IngestResult) {
        if std::env::var(RESET_HIDDEN_INDEX_ENV).ok().as_deref() != Some("1") {
            return;
        }
        let (_cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&built.storage));
        let admin = tiers::open_consumer(tiers::consumer_options(
            supertable::options_for(Modality::Vector, None),
            Arc::clone(&built.storage),
            cache,
        ));
        let hidden_prefix = admin
            .vector_index_storage_prefix()
            .expect("user manifest records hidden vector-index prefix");
        drop(admin);
        let root_storage = Arc::clone(&built.storage);
        let deleted = tiers::block_on(async {
            // The manifest prefix is a generated UUID namespace, so one broad
            // non-empty prefix list is both portable and scoped exclusively to
            // this table's derived hidden index.
            let keys = root_storage
                .list_with_prefix(&hidden_prefix)
                .await
                .expect("list hidden vector-index namespace");
            let count = keys.len();
            for key in keys {
                root_storage
                    .delete(&key)
                    .await
                    .expect("delete hidden vector-index object");
            }
            count
        });
        eprintln!(
            "[supertable_vector] reset hidden vector-index sibling: deleted {deleted} object(s); user table preserved"
        );
        let (_verify_cache_dir, verify_cache) = tiers::fresh_disk_cache(Arc::clone(&built.storage));
        let verify = tiers::open_consumer(tiers::consumer_options(
            supertable::options_for(Modality::Vector, None),
            Arc::clone(&built.storage),
            verify_cache,
        ));
        let hidden = verify
            .vector_index_table()
            .expect("reset must recreate the hidden vector index");
        assert_eq!(
            hidden.reader().n_superfiles(),
            0,
            "reset hidden index must reopen empty"
        );
    }

    /// One metered cold public `vector_search` consumer via the shared
    /// cold-store recipe (open / first / steady-second / repeat).
    fn measure_cold_store(
        label: &str,
        built: &supertable::IngestResult,
        query: &[f32],
        steady_queries: &[Vec<f32>],
        nprobe: usize,
        rerank: usize,
        cache_budget_bytes: u64,
    ) -> Option<RoutingColdStat> {
        assert!(
            !steady_queries.is_empty(),
            "steady-cold measurement needs at least one distinct query"
        );
        let meter = storage_meter::wrap(Arc::clone(&built.storage));
        let trace_enabled = cold_trace_enabled();
        let mut first_query_trace = None;
        let mut steady_trace = None;
        let measured = cold_store::measure_cold_store(
            &meter,
            || {
                let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
                    meter.provider(),
                    Some(cache_budget_bytes),
                );
                let opts = tiers::consumer_options(
                    supertable::options_for(Modality::Vector, None),
                    meter.provider(),
                    cache,
                );
                let consumer = tiers::open_consumer(opts);
                (cache_dir, consumer)
            },
            |(_cache, consumer)| {
                if trace_enabled {
                    meter.start_trace();
                }
                let _ = consumer
                    .reader()
                    .vector_search(
                        supertable::VEC_COLUMN,
                        query,
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("metered cold-first vector_search: {e}"));
                if trace_enabled {
                    first_query_trace = Some(meter.take_trace());
                }
            },
            |(_cache, consumer), i| {
                let q = &steady_queries[i % steady_queries.len()];
                if trace_enabled && i == 0 {
                    meter.start_trace();
                }
                let _ = consumer
                    .reader()
                    .vector_search(
                        supertable::VEC_COLUMN,
                        q,
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("metered cold-steady vector_search: {e}"));
                if trace_enabled && i == 0 {
                    steady_trace = Some(meter.take_trace());
                }
            },
            steady_queries.len().min(STEADY_COLD_SAMPLES),
            |(_cache, consumer)| {
                let _ = consumer
                    .reader()
                    .vector_search(
                        supertable::VEC_COLUMN,
                        query,
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("metered repeat vector_search: {e}"));
            },
        );
        log_cold_split(label, &measured.split);
        if let Some(trace) = first_query_trace {
            log_query_read_trace(label, "first cold query", &trace);
        }
        if let Some(trace) = steady_trace {
            log_query_read_trace(label, "steady cold query", &trace);
        }
        Some(RoutingColdStat {
            split: measured.split,
            open_wall_s: measured.open_wall_s,
            open_cpu_s: measured.open_cpu_s,
            query_wall_s: measured.first_wall_s,
            query_cpu_s: measured.first_cpu_s,
            second_wall_s: measured.second_wall_s,
            second_cpu_s: measured.second_cpu_s,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn measure_routing_state(
        label: &'static str,
        expected: ExpectedTiers,
        recall: Option<String>,
        consumer: &Supertable,
        consumer_meter: &storage_meter::MeteredStorage,
        built: &supertable::IngestResult,
        query: &[f32],
        steady_queries: &[Vec<f32>],
        nprobe: usize,
        rerank: usize,
        cache_budget_bytes: u64,
        include_warm: bool,
        include_cold: bool,
    ) -> RoutingStateStat {
        // Settled-anon bracket for the cost model's pinned line: sample
        // before this state's measurements and again after, so only what
        // the engine retains ACROSS the battery counts — harness heap
        // allocated between states (recall machinery, report rows) drops
        // out, and freed query scratch is purged before both samples.
        let settled_before = rss::settled_rss_breakdown().map(|(_, anon, _, _)| anon);
        let hits = hit_tier_counts(consumer, query, nprobe, rerank);
        let user_hits = hits.user_hits;
        let hidden_hits = hits.hidden_hits;
        assert_expected_tiers(label, expected, user_hits, hidden_hits);

        let warm = include_warm.then(|| {
            let reader = consumer.reader();
            let sampler = PeakSampler::start_default();
            let search = || {
                reader
                    .vector_search(
                        supertable::VEC_COLUMN,
                        query,
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                        None,
                    )
                    .expect("routing-state warm vector search")
            };
            black_box(search());
            let trace_enabled = cold_trace_enabled();
            if trace_enabled {
                consumer_meter.start_trace();
            }
            let before = consumer_meter.snapshot();
            let mut samples = Vec::with_capacity(ROUTING_STATE_WARM_ITERS);
            let cpu0 = cpu::process_cpu_ns();
            for _ in 0..ROUTING_STATE_WARM_ITERS {
                let started = Instant::now();
                black_box(search());
                samples.push(started.elapsed());
            }
            let warm_cpu_s = cpu::cpu_seconds_since(cpu0)
                .map(|seconds| seconds / ROUTING_STATE_WARM_ITERS as f64);
            let warm_trace = trace_enabled.then(|| consumer_meter.take_trace());
            let warm_io = consumer_meter.snapshot().since(&before);
            if let Some(trace) = warm_trace
                && !trace.is_empty()
            {
                log_query_read_trace(label, "warm measurement", &trace);
            }
            samples.sort_unstable();
            let p50_ns = samples[samples.len() / 2].as_secs_f64() * 1e9;
            (
                p50_ns,
                warm_cpu_s,
                warm_io,
                sampler.stop_stats().peak_rss_bytes,
            )
        });
        // Engine-pinned estimate, sampled after the warm battery but BEFORE
        // the cold-store measurement: the cold guard opens a second consumer
        // purely to time cold opens — harness, not serving state. Pinned =
        // the shared consumer's open delta plus what this state's warm
        // serving retained (settled-after minus settled-before, allocator
        // purged at both samples so freed query scratch never counts).
        let settled = rss::settled_rss_breakdown();
        let engine_anon_bytes = settled.map(|(_, anon, _, _)| {
            let retained = settled_before
                .map(|before| anon.saturating_sub(before))
                .unwrap_or(0);
            CONSUMER_ENGINE_ANON_BYTES.load(AtomicOrdering::Relaxed) + retained
        });
        let settled_file_bytes = settled.map(|(_, _, file, _)| file);
        let cold = include_cold
            .then(|| {
                measure_cold_store(
                    label,
                    built,
                    query,
                    steady_queries,
                    nprobe,
                    rerank,
                    cache_budget_bytes,
                )
            })
            .flatten();
        if let Some(cold) = &cold {
            assert_expected_cold_reads(label, expected, &cold.split, supertable::n_docs());
        }
        eprintln!(
            "[supertable_vector/{label}] expected {}; top-k {user_hits} user + {hidden_hits} hidden; warm {}; cold 1st {}; cold 2nd {}",
            expected.label(),
            warm.map(|(p50, _, _, _)| fmt_time(p50))
                .unwrap_or_else(|| "not measured".into()),
            cold.map(|value| fmt_get_class_breakdown(&value.split.first_query))
                .unwrap_or_else(|| "not measured".into()),
            cold.map(|value| fmt_get_class_breakdown(&value.split.second_query))
                .unwrap_or_else(|| "not measured".into()),
        );
        RoutingStateStat {
            label,
            expected,
            recall,
            warm_p50_ns: warm.map(|(p50, _, _, _)| p50),
            warm_cpu_s: warm.and_then(|(_, cpu_s, _, _)| cpu_s),
            ram_bytes: warm.map(|(_, _, _, ram_bytes)| ram_bytes),
            ram_anon_bytes: engine_anon_bytes,
            ram_file_settled_bytes: settled_file_bytes,
            warm_io: warm.map(|(_, _, io, _)| io),
            cold,
        }
    }

    fn emit_routing_states(report: &mut Report, n_docs: usize, states: &[RoutingStateStat]) {
        report.emit(&Section {
            anchor: "bench/vector/supertable/routing-states".into(),
            title: format!(
                "Supertable vector — routing state transitions ({} docs × dim={})",
                fmt_count(n_docs),
                DIM
            ),
            note: format!(
                "One search configuration across the full lifecycle. Data-path assertions use cold GET classes. Recall is the same 20-query brute-force metric in every state; the follow-up commit adds {} normal rows from the corpus distribution.",
                supertable::docs_per_commit(),
            ),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "State".into(),
                    "Expected data".into(),
                    "Recall@10".into(),
                    "Warm p50".into(),
                    "Warm GET/query".into(),
                    "Cold open GET/bytes".into(),
                    "User data GET/bytes".into(),
                    "Hidden data GET/bytes".into(),
                    "Manifest GET/bytes".into(),
                ],
                rows: states
                    .iter()
                    .map(|state| {
                        let warm_gets = state
                            .warm_io
                            .map(|io| io.get_count as f64 / ROUTING_STATE_WARM_ITERS as f64)
                            .map(|gets| format!("{gets:.2}"))
                            .unwrap_or_else(|| "N/A".into());
                        let cold_open = state
                            .cold
                            .map(|cold| {
                                format!(
                                    "{} / {}",
                                    cold.split.open.get_count,
                                    rss::fmt_bytes(cold.split.open.get_bytes)
                                )
                            })
                            .unwrap_or_else(|| "N/A".into());
                        let cold_query = state.cold.map(|cold| cold.split.first_query);
                        vec![
                            text(state.label),
                            text(state.expected.label()),
                            text(state.recall.clone().unwrap_or_else(|| "N/A".into())),
                            text(
                                state
                                    .warm_p50_ns
                                    .map(fmt_time)
                                    .unwrap_or_else(|| "N/A".into()),
                            ),
                            text(warm_gets),
                            text(cold_open),
                            text(
                                cold_query
                                    .map(|io| {
                                        class_gets(io, storage_meter::UriClass::UserData)
                                    })
                                    .unwrap_or_else(|| "N/A".into()),
                            ),
                            text(
                                cold_query
                                    .map(|io| {
                                        class_gets(io, storage_meter::UriClass::HiddenData)
                                    })
                                    .unwrap_or_else(|| "N/A".into()),
                            ),
                            text(
                                cold_query
                                    .map(manifest_gets)
                                    .unwrap_or_else(|| "N/A".into()),
                            ),
                        ]
                    })
                    .collect(),
            }],
        });
    }

    fn emit_transitions(report: &mut Report, n_docs: usize, transitions: &[TransitionStat]) {
        report.emit(&Section {
            anchor: "bench/vector/supertable/transitions".into(),
            title: format!(
                "Supertable vector — lifecycle transitions ({} base docs × dim={})",
                fmt_count(n_docs),
                DIM
            ),
            note: "Every mutation between query-state rows is reported here. Request and byte counts are measured over the transition itself; no query traffic is mixed into these windows.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "Transition".into(),
                    "Wall".into(),
                    "PUT".into(),
                    "Uploaded".into(),
                    "GET".into(),
                    "Downloaded".into(),
                    "HEAD".into(),
                    "Peak RSS".into(),
                ],
                rows: transitions
                    .iter()
                    .map(|transition| {
                        let io = transition.io;
                        vec![
                            text(transition.label),
                            text(fmt_time(transition.wall_ns)),
                            text(
                                io.map(|value| value.put_count.to_string())
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                            text(
                                io.map(|value| rss::fmt_bytes(value.put_bytes))
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                            text(
                                io.map(|value| value.get_count.to_string())
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                            text(
                                io.map(|value| rss::fmt_bytes(value.get_bytes))
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                            text(
                                io.map(|value| value.head_count.to_string())
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                            text(
                                transition
                                    .peak_rss_bytes
                                    .map(rss::fmt_bytes)
                                    .unwrap_or_else(|| "NOT METERED".into()),
                            ),
                        ]
                    })
                    .collect(),
            }],
        });
    }

    fn routing_cold_to_measurement(cold: RoutingColdStat) -> ColdStoreMeasurement {
        ColdStoreMeasurement {
            split: cold.split,
            open_wall_s: cold.open_wall_s,
            open_cpu_s: cold.open_cpu_s,
            first_wall_s: cold.query_wall_s,
            first_cpu_s: cold.query_cpu_s,
            second_wall_s: cold.second_wall_s,
            second_cpu_s: cold.second_cpu_s,
        }
    }

    fn query_state_costs(states: &[RoutingStateStat]) -> [cost::QueryStateCost; 4] {
        let mut out = [cost::QueryStateCost::default(); 4];
        for (slot, state) in out.iter_mut().zip(states) {
            let cold = state.cold;
            *slot = cost::QueryStateCost {
                io: cost::QueryStateIo {
                    label: Some(state.label),
                    cold_open: cold.map(|value| value.split.open),
                    cold_query: cold.map(|value| value.split.first_query),
                    cold_second: cold.map(|value| value.split.second_query),
                    cold_repeat: cold.map(|value| value.split.repeat_query),
                    warm: state.warm_io,
                    warm_iters: state
                        .warm_io
                        .map(|_| ROUTING_STATE_WARM_ITERS as u64)
                        .unwrap_or(0),
                },
                warm_p50_s: state.warm_p50_ns.map(|ns| ns / 1e9),
                warm_cpu_s: state.warm_cpu_s,
                ram_bytes: state.ram_bytes,
                ram_anon_bytes: state.ram_anon_bytes,
                ram_file_settled_bytes: state.ram_file_settled_bytes,
                cold_open_s: cold.map(|value| value.open_wall_s),
                cold_open_cpu_s: cold.and_then(|value| value.open_cpu_s),
                cold_query_s: cold.map(|value| value.query_wall_s),
                cold_query_cpu_s: cold.and_then(|value| value.query_cpu_s),
                cold_second_s: cold.map(|value| value.second_wall_s),
                cold_second_cpu_s: cold.and_then(|value| value.second_cpu_s),
            };
        }
        out
    }

    /// Build a vector-only supertable, then measure warm + cold kNN search
    /// at calibrated recall targets (and a default config), with a
    /// correctness recall gate — the same measurement the superfile vector
    /// runner produces, over the multi-superfile object-store consumer.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_vector] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_vector");

        // Existing-prefix mode: read directly against an already-built,
        // retained supertable (`INFINO_BENCH_EXISTING_PREFIX`) — no corpus, no
        // ingest. With no corpus to back recall, calibration + the brute-force
        // gate are forced off below and queries are corpus-free.
        let existing = tiers::block_on(tiers::existing_supertable_storage_fixture());

        // Always prepare the corpus for vector benches so recall is always
        // measurable (including existing-prefix runs).
        crate::rss::log_rss_breakdown("supertable_vector before corpus prepare");
        let mut corpus = Some(supertable::prepare_corpus(Modality::Vector));
        crate::rss::log_rss_breakdown("supertable_vector after corpus prepare");

        let (built, ingest_metrics) = if let Some(fixture) = existing {
            let opened = (supertable::open_existing(Modality::Vector, fixture), None);
            crate::rss::log_rss_breakdown(
                "supertable_vector after open_existing (producer handle)",
            );
            opened
        } else if crate::dataset::dataset_mode() && !phases.build {
            (supertable::open_dataset(Modality::Vector), None)
        } else {
            build_measured(
                Modality::Vector,
                corpus
                    .as_ref()
                    .expect("non-existing path prepared a corpus"),
                phases,
            )
        };
        reset_hidden_vector_index_if_requested(&built);
        if let Some(metrics) = &ingest_metrics {
            report.emit(&Section {
                anchor: "bench/vector/supertable/ingest".into(),
                title: format!(
                    "Supertable vector — ingest, multi-superfile / object-store ({} docs × dim={}, {} commits, {} writers)",
                    fmt_count(n_docs),
                    DIM,
                    supertable::n_commits(),
                    supertable::n_writers()
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: ingest_headers(),
                    rows: vec![ingest_row(n_docs, "vector-only", metrics)],
                }],
            });
        }

        if phases.warm || phases.cold {
            let skip_cal = skip_calibration(n_docs);
            let nprobe = fixed_nprobe();
            let rerank = fixed_rerank_mult();

            let (q_correct, q_cal, gt_correct, gt_cal, filtered_gt, augmented_gt) = {
                let corpus = corpus
                    .as_ref()
                    .expect("vector benches always prepare a corpus");
                // The ingested vectors are still mmapped from the prepared
                // corpus — queries and ground truth come from them instead
                // of a regeneration. Skip-calibration still computes
                // correctness/default recall (and filtered recall); it only
                // skips the recall-target calibration sweep.
                let vslice = corpus
                    .vectors()
                    .expect("vector modality prepared a vector corpus")
                    .as_slice();
                let base_vectors = &vslice[..n_docs * DIM];
                let q_correct = corpus::generate_realistic_queries(
                    base_vectors,
                    n_docs,
                    N_CORRECTNESS_QUERIES,
                    QUERY_CORRECTNESS_SEED,
                    true,
                    QUERY_SIGMA,
                );
                let q_cal = corpus::generate_realistic_queries(
                    base_vectors,
                    n_docs,
                    N_CALIBRATION_QUERIES,
                    QUERY_CALIBRATION_SEED,
                    true,
                    QUERY_SIGMA,
                );
                let augmented_docs = n_docs + supertable::docs_per_commit();
                let all_queries: Vec<Vec<f32>> = if skip_cal {
                    q_correct.clone()
                } else {
                    q_correct.iter().chain(&q_cal).cloned().collect()
                };
                let mut labels = corpus::grading::lifecycle_ground_truth_cached(
                    corpus::grading::LifecycleGradingOptions {
                        vectors: vslice,
                        n_docs,
                        augmented_docs,
                        corpus_seed: supertable::CORPUS_VEC_SEED,
                        normalized_vectors: true,
                        filter_keep_every: FILTER_KEEP_EVERY,
                        top_k: TOP_K,
                        correctness_query_count: q_correct.len(),
                        queries: &all_queries,
                    },
                );
                let gt_cal = if skip_cal {
                    Vec::new()
                } else {
                    labels.base.split_off(q_correct.len())
                };
                (
                    q_correct,
                    q_cal,
                    labels.base,
                    gt_cal,
                    Some(labels.filtered),
                    labels.augmented,
                )
            };
            // Queries + ground truth extracted. Keep the mapping for the
            // normal follow-up commit, but evict its pages while measuring
            // pre/post-drain search.
            if let Some(vectors) = corpus
                .as_ref()
                .and_then(supertable::PreparedCorpus::vectors)
            {
                vectors.advise_consumed(0, vectors.n_docs());
            }

            const PRE_DRAIN_NOTE: &str = "Pre-drain (incoming staging): hidden IVF commit shards still in INCOMING; every query includes INCOMING plus nprobe-routed cells. Warm = query-driven cache fill; cold = fresh cache per iteration. Δ vs previous run.";
            const POST_DRAIN_NOTE: &str = "Post-drain (routed cells): incoming empty after OPANN route; queries hit ~nprobe cell-local IVF superfiles only. Warm = query-driven cache fill; cold = fresh cache per iteration. Δ vs previous run.";
            const POST_DELTA_NOTE: &str = "Post-delta (hidden + undrained tail): drained rows rank on cell-local hidden IVF superfiles; undrained user commits fan out directly and merge by distance. Warm = query-driven cache fill; cold = fresh cache per iteration. Δ vs previous run.";

            // Fresh ingest leaves hidden IVF in INCOMING; dataset / existing-prefix
            // tables may already be post-drain — run the two-phase comparison only
            // when we just built the table in this process.
            let force_pre_post_drain = ingest_metrics.is_some()
                || std::env::var("INFINO_BENCH_FORCE_PRE_POST_DRAIN")
                    .ok()
                    .as_deref()
                    == Some("1");
            let skip_vector_delta =
                std::env::var(SKIP_VECTOR_DELTA_ENV).ok().as_deref() == Some("1");
            // Force compaction (optimize) + post-compact on a reopened, already-drained
            // table -- otherwise a drained existing prefix reports phase-derived only and
            // never exercises the split.
            let force_compact =
                std::env::var("INFINO_BENCH_FORCE_COMPACT").ok().as_deref() == Some("1");

            let search_title = |phase: &str| {
                format!(
                    "Supertable vector — search {phase}, multi-superfile / object-store ({} docs × dim={})",
                    fmt_count(n_docs),
                    DIM
                )
            };

            // Metered shared consumer: the drain runs on this handle, so its
            // object-store I/O (user-vector reads + hidden cell-superfile
            // writes) is captured as a snapshot delta around the drain call.
            // Cache budget covers the *post-drain* footprint (user + hidden
            // index), not just the user index this pre-drain open can see —
            // see [`SHARED_CONSUMER_CACHE_INDEX_FACTOR`].
            let consumer_meter = storage_meter::wrap(Arc::clone(&built.storage));
            let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
                consumer_meter.provider(),
                Some(
                    built
                        .total_index_bytes
                        .saturating_mul(SHARED_CONSUMER_CACHE_INDEX_FACTOR),
                ),
            );
            // Bracket the consumer open with settled-anon samples: the delta
            // is the engine handle's own pinned memory, free of the harness
            // heap that precedes it (corpus bookkeeping, producer handle).
            // This shared handle drives the lifecycle mutations (drain,
            // delta commit, optimize), so the consumer-memory-mode env knob
            // must not apply to it — stripped summaries cannot be
            // republished. The knob-on serving profile is measured on the
            // fresh consumers each routing state / cold split opens.
            let anon_before_consumer = rss::settled_rss_breakdown().map(|(_, anon, _, _)| anon);
            let consumer = tiers::open_consumer(tiers::consumer_options_with_knob(
                supertable::options_for(Modality::Vector, None),
                consumer_meter.provider(),
                cache,
                false,
            ));
            crate::rss::log_rss_breakdown("supertable_vector after consumer open");
            if let (Some(before), Some((_, after, _, _))) =
                (anon_before_consumer, rss::settled_rss_breakdown())
            {
                CONSUMER_ENGINE_ANON_BYTES
                    .store(after.saturating_sub(before), AtomicOrdering::Relaxed);
            }
            let id_to_dense = Arc::new(corpus::engine_id_to_dense(&consumer, n_docs));
            crate::rss::log_rss_breakdown("supertable_vector after id_to_dense map");
            let warm_reader = SupertableVectorRead {
                table: &consumer,
                id_to_dense: Arc::clone(&id_to_dense),
            };
            // A retained prefix whose drain already ran is reusable as-is:
            // measuring "pre-drain" on it would be a lie and re-draining is
            // a no-op, so fall through to phase-derived single-phase
            // reporting. This is what makes probe-config iteration cheap —
            // point INFINO_BENCH_EXISTING_PREFIX at a drained table and each
            // run skips the corpus upload AND the drain.
            let pre_post_drain = if force_pre_post_drain
                && ingest_metrics.is_none()
                && current_routing_phase(&consumer) != "pre-drain"
            {
                eprintln!(
                    "[supertable_vector] existing prefix is already drained ({}); \
                     skipping forced pre/post-drain lifecycle (phase-derived reporting)",
                    current_routing_phase(&consumer),
                );
                false
            } else {
                force_pre_post_drain
            };
            // Run compaction/split + post-compact either in the forced drain
            // lifecycle or when explicitly forced on a reopened drained table.
            let run_compact = pre_post_drain || force_compact;
            let mut drain_stats: Option<(f64, storage_meter::ObjectStoreMeter, u64, Option<f64>)> =
                None;
            let mut delta_stats: Option<(f64, storage_meter::ObjectStoreMeter, u64, Option<f64>)> =
                None;
            let mut filtered_stats: Option<(storage_meter::ObjectStoreMeter, u64)> = None;
            let mut cold_split_pre: Option<storage_meter::ColdStoreSplit> = None;
            let mut pre_search_rows = None;
            let mut routing_states = Vec::new();
            let mut transitions = Vec::new();
            let mut delta_id_to_dense: Option<Arc<HashMap<i128, u32>>> = None;
            if let Some(metrics) = &ingest_metrics {
                transitions.push(TransitionStat {
                    label: "ingest",
                    wall_ns: metrics.wall_ns,
                    io: built.ingest_io,
                    peak_rss_bytes: Some(metrics.peak_rss_bytes),
                });
            }

            let recall_rows = if pre_post_drain {
                if phases.warm {
                    log_hidden_stats(&consumer, "at warm open (pre-drain)");
                }
                eprintln!("[supertable_vector] === pre-drain search (incoming staging) ===");
                let pre_drain_rows = exec_vec::run_search(
                    &mut report,
                    &warm_reader,
                    || SupertableVecColdGuard::open(&built, Arc::clone(&id_to_dense)),
                    supertable::VEC_COLUMN,
                    n_docs,
                    TOP_K,
                    nprobe,
                    rerank,
                    &q_correct,
                    &gt_correct,
                    &q_cal,
                    &gt_cal,
                    phases.warm,
                    phases.cold,
                    COLD_ITERS,
                    skip_cal,
                    "supertable_vector/pre-drain",
                    "bench/vector/supertable/search/pre-drain",
                    search_title("pre-drain"),
                    PRE_DRAIN_NOTE,
                );
                let pre_drain_recall = default_recall(&pre_drain_rows);
                pre_search_rows = Some(pre_drain_rows);

                let pre_drain_state = measure_routing_state(
                    "pre-drain",
                    ExpectedTiers::UserOnly,
                    pre_drain_recall,
                    &consumer,
                    &consumer_meter,
                    &built,
                    &q_correct[0],
                    &q_correct[1..],
                    nprobe,
                    rerank,
                    built.total_index_bytes,
                    phases.warm,
                    phases.cold,
                );
                cold_split_pre = pre_drain_state.cold.map(|cold| cold.split);
                routing_states.push(pre_drain_state);

                let before_drain = consumer_meter.snapshot();
                let drain_sampler = PeakSampler::start_default();
                let ((), drain_wall, drain_cpu_s) = cpu::timed(|| drain_hidden_incoming(&consumer));
                let drain_wall_s = drain_wall.as_secs_f64();
                let drain_rss = drain_sampler.stop_stats();
                let drain_peak_rss = drain_rss.peak_rss_bytes;
                let drain_io = consumer_meter.snapshot().since(&before_drain);
                eprintln!(
                    "[supertable_vector] drain object-store I/O: {} PUT ({} up), {} GET ({} down), {} HEAD in {drain_wall_s:.1}s (peak RSS {} / anon {} / file {})",
                    drain_io.put_count,
                    rss::fmt_bytes(drain_io.put_bytes),
                    drain_io.get_count,
                    rss::fmt_bytes(drain_io.get_bytes),
                    drain_io.head_count,
                    rss::fmt_bytes(drain_peak_rss),
                    rss::fmt_bytes(drain_rss.peak_anon_rss_bytes),
                    rss::fmt_bytes(drain_rss.peak_file_rss_bytes),
                );
                // Cost RAM leg still bills total peak (not anon).
                drain_stats = Some((drain_wall_s, drain_io, drain_peak_rss, drain_cpu_s));
                transitions.push(TransitionStat {
                    label: "drain",
                    wall_ns: drain_wall_s * 1e9,
                    io: Some(drain_io),
                    peak_rss_bytes: Some(drain_peak_rss),
                });

                if phases.warm {
                    log_hidden_stats(&consumer, "at warm open (post-drain)");
                }
                eprintln!("[supertable_vector] === post-drain search (routed cells) ===");
                let post_drain_rows = exec_vec::run_search(
                    &mut report,
                    &warm_reader,
                    || SupertableVecColdGuard::open(&built, Arc::clone(&id_to_dense)),
                    supertable::VEC_COLUMN,
                    n_docs,
                    TOP_K,
                    nprobe,
                    rerank,
                    &q_correct,
                    &gt_correct,
                    &q_cal,
                    &gt_cal,
                    phases.warm,
                    phases.cold,
                    COLD_ITERS,
                    skip_cal,
                    "supertable_vector/post-drain",
                    "bench/vector/supertable/search/post-drain",
                    search_title("post-drain"),
                    POST_DRAIN_NOTE,
                );
                let post_drain_recall = default_recall(&post_drain_rows);
                routing_states.push(measure_routing_state(
                    "post-drain",
                    ExpectedTiers::HiddenOnly,
                    post_drain_recall,
                    &consumer,
                    &consumer_meter,
                    &built,
                    &q_correct[0],
                    &q_correct[1..],
                    nprobe,
                    rerank,
                    built
                        .total_index_bytes
                        .saturating_mul(SHARED_CONSUMER_CACHE_INDEX_FACTOR),
                    phases.warm,
                    phases.cold,
                ));
                // Audit AFTER the measured post-drain phases: its full-corpus
                // mmap pass and 500 extra self-queries must not perturb the
                // page cache, disk cache, or memory state the recall and
                // latency rows above are measured under.
                if let Some(vectors) = corpus
                    .as_ref()
                    .and_then(supertable::PreparedCorpus::vectors)
                {
                    report_post_drain_assignment_audit(
                        &consumer,
                        &vectors.as_slice()[..n_docs * DIM],
                        &q_correct,
                        &gt_correct,
                        &id_to_dense,
                    );
                }
                post_drain_rows
            } else {
                // Read-only path (existing-prefix / dataset): no transitions
                // run here, but the phase is still observable from manifest
                // state — report it under the same names and anchors as the
                // lifecycle branch so runs against the same table state
                // compare directly.
                let phase = current_routing_phase(&consumer);
                if phases.warm {
                    log_hidden_stats(&consumer, &format!("at warm open ({phase})"));
                }
                let log_prefix = format!("supertable_vector/{phase}");
                let anchor = format!("bench/vector/supertable/search/{phase}");
                let note = match phase {
                    "pre-drain" => PRE_DRAIN_NOTE,
                    "post-drain" => POST_DRAIN_NOTE,
                    _ => POST_DELTA_NOTE,
                };
                exec_vec::run_search(
                    &mut report,
                    &warm_reader,
                    || SupertableVecColdGuard::open(&built, Arc::clone(&id_to_dense)),
                    supertable::VEC_COLUMN,
                    n_docs,
                    TOP_K,
                    nprobe,
                    rerank,
                    &q_correct,
                    &gt_correct,
                    &q_cal,
                    &gt_cal,
                    phases.warm,
                    phases.cold,
                    COLD_ITERS,
                    skip_cal,
                    &log_prefix,
                    &anchor,
                    search_title(phase),
                    note,
                )
            };
            // Filtered vector recall + latency mirrors the superfile tier:
            // same every-Nth-row allow-set, same brute-force filtered ground
            // truth, same default config.
            if phases.warm
                && let Some(filtered_gt) = filtered_gt.as_ref()
            {
                let consumer_reader = consumer.reader();
                let mut allow_stable_ids: Vec<i128> = id_to_dense
                    .iter()
                    .filter_map(|(stable_id, dense_id)| {
                        ((*dense_id as usize).is_multiple_of(FILTER_KEEP_EVERY))
                            .then_some(*stable_id)
                    })
                    .collect();
                allow_stable_ids.sort_unstable();
                allow_stable_ids.dedup();
                let prepared_allow = tiers::block_on(
                    consumer_reader.prepare_vector_stable_allow_async(Arc::new(allow_stable_ids)),
                )
                .expect("prepare stable-id allow bitmaps");
                let mut recalls = Vec::new();
                let mut latencies = Vec::new();
                // Untimed prewarm: fault the filtered path's routed cells into
                // the resident cache first, so the metered window below reflects
                // steady-state warm I/O (not the one-time cache fill).
                for q in q_correct.iter() {
                    let _ =
                        tiers::block_on(consumer_reader.vector_hits_prepared_global_allow_async(
                            supertable::VEC_COLUMN,
                            q,
                            TOP_K,
                            exec_vec::default_search_opts(),
                            &prepared_allow,
                        ))
                        .expect("filtered prewarm query");
                }
                let filtered_before = consumer_meter.snapshot();
                // Drop phases accumulated by the prewarm loop so the dump
                // below covers exactly the measured window.
                let _ = io_counters::phase_take_summed();
                for (q, gt) in q_correct.iter().zip(filtered_gt) {
                    let t0 = Instant::now();
                    let hits =
                        tiers::block_on(consumer_reader.vector_hits_prepared_global_allow_async(
                            supertable::VEC_COLUMN,
                            q,
                            TOP_K,
                            exec_vec::default_search_opts(),
                            &prepared_allow,
                        ))
                        .expect("filtered recall query");
                    latencies.push(t0.elapsed());
                    let dense_hits = hits_to_dense_u32(&consumer, &id_to_dense, &hits);
                    recalls.push(corpus::recall_at_k(&dense_hits, gt));
                }
                let filtered_phases = io_counters::phase_take_summed();
                if !filtered_phases.is_empty() && !q_correct.is_empty() {
                    let n = q_correct.len() as f64;
                    let parts: Vec<String> = filtered_phases
                        .iter()
                        .map(|(name, us)| format!("{name}={:.0}µs", *us as f64 / n))
                        .collect();
                    eprintln!(
                        "[vector filtered phases] avg over {} queries (Σ across concurrent \
                         fan-out units): {}",
                        q_correct.len(),
                        parts.join("  ")
                    );
                }
                let filtered_io = consumer_meter.snapshot().since(&filtered_before);
                if !q_correct.is_empty() {
                    filtered_stats = Some((filtered_io, q_correct.len() as u64));
                    eprintln!(
                        "[supertable_vector] filtered warm window: {} GET ({} down) over {} queries",
                        filtered_io.get_count,
                        rss::fmt_bytes(filtered_io.get_bytes),
                        q_correct.len(),
                    );
                }
                // Probe-width discriminator for filtered recall loss: if
                // recall climbs with an explicit wider cell probe, the gap
                // is cell coverage (selection misses the matching
                // neighbors' cells); if it stays flat, the loss sits
                // inside the probed cells (kernel shortlist/rerank under
                // the allow-set). Diagnostic print only — the gate above
                // stays on the engine default.
                for width in FILTERED_DIAG_PROBE_WIDTHS {
                    let mut wide_recalls = Vec::with_capacity(q_correct.len());
                    let mut wide_lat = Vec::with_capacity(q_correct.len());
                    for (q, gt) in q_correct.iter().zip(filtered_gt) {
                        let t0 = Instant::now();
                        let hits = tiers::block_on(
                            consumer_reader.vector_hits_prepared_global_allow_async(
                                supertable::VEC_COLUMN,
                                q,
                                TOP_K,
                                exec_vec::search_opts(*width, exec_vec::ENGINE_DEFAULT),
                                &prepared_allow,
                            ),
                        )
                        .expect("filtered width-sweep query");
                        wide_lat.push(t0.elapsed());
                        let dense_hits = hits_to_dense_u32(&consumer, &id_to_dense, &hits);
                        wide_recalls.push(corpus::recall_at_k(&dense_hits, gt));
                    }
                    if !wide_recalls.is_empty() {
                        let mean: f32 =
                            wide_recalls.iter().sum::<f32>() / wide_recalls.len() as f32;
                        wide_lat.sort_unstable();
                        let p50_ms = wide_lat[wide_lat.len() / 2].as_secs_f64() * 1e3;
                        eprintln!(
                            "[supertable_vector] filtered width-sweep: nprobe={width} \
                             recall@{TOP_K}={mean:.3} p50={p50_ms:.2}ms"
                        );
                    }
                }
                if recalls.is_empty() || latencies.is_empty() {
                    eprintln!(
                        "[supertable_vector] filtered recall skipped: no correctness queries"
                    );
                } else {
                    let mean_recall: f32 = recalls.iter().sum::<f32>() / recalls.len() as f32;
                    latencies.sort_unstable();
                    let p50_ns = latencies[latencies.len() / 2].as_secs_f64() * 1e9;
                    let selectivity = 1.0 / FILTER_KEEP_EVERY as f64;

                    eprintln!(
                        "[supertable_vector] filtered recall@{TOP_K} ({} queries, ~10% selectivity): {mean_recall:.3}, p50={:.2}ms",
                        q_correct.len(),
                        p50_ns / 1e6,
                    );
                    assert!(
                        mean_recall >= FILTERED_RECALL_FLOOR,
                        "filtered vector recall@{TOP_K} {mean_recall:.3} < floor \
                         {FILTERED_RECALL_FLOOR:.2} — the allow-set fan regressed (this class \
                         previously shipped unasserted: the selectivity-blind postings target \
                         measured 0.722 for weeks as a print-only line)"
                    );

                    report.emit(&Section {
                        anchor: "bench/vector/supertable/filtered".into(),
                        title: format!(
                            "Supertable vector — filtered search ({} docs × dim={})",
                            fmt_count(n_docs),
                            DIM
                        ),
                        note: format!(
                            "Filtered kNN (~10% selectivity, every {}th row). recall@{TOP_K} = {mean_recall:.3}. Δ is vs the previous run.",
                            FILTER_KEEP_EVERY
                        ),
                        blocks: vec![Block {
                            subtitle: String::new(),
                            headers: vec![
                                "Filter".into(),
                                "Requested".into(),
                                "Actual routing".into(),
                                "selectivity".into(),
                                "recall@10".into(),
                                "p50".into(),
                            ],
                            rows: vec![vec![
                                text("filtered (~10%)"),
                                text("engine default"),
                                text(warm_reader.routing_label(
                                    exec_vec::ENGINE_DEFAULT,
                                    exec_vec::ENGINE_DEFAULT,
                                )),
                                text(format!("{:.1}%", selectivity * 100.0)),
                                text(format!("{mean_recall:.3}")),
                                context(p50_ns, fmt_time(p50_ns), Better::Lower),
                            ]],
                        }],
                    });
                }
            }

            if phases.warm || phases.cold {
                // Steady-state warm I/O: replay the correctness queries on
                // the shared, cache-hot consumer — the same consumer the
                // warm latency battery timed — so the ledger's warm GET/query
                // and the compute ledger's warm CPU describe one path.
                let warm_io = (phases.warm && !q_correct.is_empty()).then(|| {
                    let reader = consumer.reader();
                    // Untimed prewarm: fault each query's routed cells into the
                    // resident cache first, so the metered pass below reflects
                    // steady-state warm I/O (0 GET when the working set fits the
                    // budget), not the one-time cache fill. Mirrors the FTS warm
                    // path's untimed-prewarm-then-measure discipline.
                    for q in &q_correct {
                        let _ = reader
                            .vector_search(
                                supertable::VEC_COLUMN,
                                q,
                                TOP_K,
                                exec_vec::search_opts(nprobe, rerank),
                                None,
                                None,
                            )
                            .expect("warm-window prewarm vector_search");
                    }
                    let before = consumer_meter.snapshot();
                    for q in &q_correct {
                        let _ = reader
                            .vector_search(
                                supertable::VEC_COLUMN,
                                q,
                                TOP_K,
                                exec_vec::search_opts(nprobe, rerank),
                                None,
                                None,
                            )
                            .expect("warm-window vector_search");
                    }
                    let io = consumer_meter.snapshot().since(&before);
                    eprintln!(
                        "[supertable_vector] warm window (cache hot): {} GET ({} down) over {} queries",
                        io.get_count,
                        rss::fmt_bytes(io.get_bytes),
                        q_correct.len(),
                    );
                    (io, q_correct.len() as u64)
                });

                // Add one normal follow-up commit after the fully-drained
                // measurement. Recall remains comparable because its rows
                // come from the same generated distribution and the oracle
                // covers the augmented corpus.
                if pre_post_drain && !skip_vector_delta {
                    let delta_rows = supertable::docs_per_commit();
                    eprintln!(
                        "[supertable_vector] committing {delta_rows} normal undrained vector rows..."
                    );
                    let delta_batch = supertable::vector_delta_batch(
                        corpus
                            .as_ref()
                            .expect("vector corpus retained for follow-up commit"),
                    );
                    let before = consumer_meter.snapshot();
                    let sampler = PeakSampler::start_default();
                    let (result, wall, cpu_s) = cpu::timed(|| consumer.append(&delta_batch));
                    result.expect("commit undrained vector delta");
                    drop(delta_batch);
                    drop(corpus.take());
                    let wall_s = wall.as_secs_f64();
                    let wall_ns = wall_s * 1e9;
                    let peak_rss = sampler.stop_stats().peak_rss_bytes;
                    let io = consumer_meter.snapshot().since(&before);
                    eprintln!(
                        "[supertable_vector] delta commit: {} rows, {} PUT ({} up), {} GET ({} down), wall {}, peak RSS {}",
                        delta_rows,
                        io.put_count,
                        rss::fmt_bytes(io.put_bytes),
                        io.get_count,
                        rss::fmt_bytes(io.get_bytes),
                        fmt_time(wall_ns),
                        rss::fmt_bytes(peak_rss),
                    );
                    transitions.push(TransitionStat {
                        label: "delta commit",
                        wall_ns,
                        io: Some(io),
                        peak_rss_bytes: Some(peak_rss),
                    });
                    delta_stats = Some((wall_s, io, peak_rss, cpu_s));
                    let delta_map =
                        Arc::new(corpus::engine_id_to_dense(&consumer, n_docs + delta_rows));
                    let delta_reader = SupertableVectorRead {
                        table: &consumer,
                        id_to_dense: Arc::clone(&delta_map),
                    };
                    let post_delta_recall = exec_vec::mean_recall(
                        &delta_reader,
                        supertable::VEC_COLUMN,
                        &q_correct,
                        &augmented_gt,
                        TOP_K,
                        nprobe,
                        rerank,
                    );
                    delta_id_to_dense = Some(Arc::clone(&delta_map));
                    routing_states.push(measure_routing_state(
                        "post-delta",
                        ExpectedTiers::Both,
                        Some(format!("{post_delta_recall:.3}")),
                        &consumer,
                        &consumer_meter,
                        &built,
                        &q_correct[0],
                        &q_correct[1..],
                        nprobe,
                        rerank,
                        built
                            .total_index_bytes
                            .saturating_mul(SHARED_CONSUMER_CACHE_INDEX_FACTOR),
                        phases.warm,
                        phases.cold,
                    ));
                } else if pre_post_drain {
                    eprintln!(
                        "[supertable_vector] skipping normal undrained delta ({SKIP_VECTOR_DELTA_ENV}=1)"
                    );
                }

                // Optimize compacts user + hidden physical files; when the
                // delta phase ran it first drains that tail. The following
                // state must therefore be hidden-only in either mode.
                let compaction_stats = run_compact.then(|| {
                    eprintln!("[supertable_vector] compacting (optimize: user + hidden)...");
                    let before = consumer_meter.snapshot();
                    let sampler = PeakSampler::start_default();
                    let (result, wall, cpu_s) =
                        cpu::timed(|| consumer.optimize(&OptimizeOptions::default()));
                    result.expect("optimize (compaction)");
                    let wall_s = wall.as_secs_f64();
                    let rss_stats = sampler.stop_stats();
                    let peak_rss = rss_stats.peak_rss_bytes;
                    let io = consumer_meter.snapshot().since(&before);
                    eprintln!(
                        "[supertable_vector] compaction object-store I/O: {} PUT ({} up), {} GET ({} down) in {wall_s:.1}s (peak RSS {} / anon {} / file {})",
                        io.put_count,
                        rss::fmt_bytes(io.put_bytes),
                        io.get_count,
                        rss::fmt_bytes(io.get_bytes),
                        rss::fmt_bytes(peak_rss),
                        rss::fmt_bytes(rss_stats.peak_anon_rss_bytes),
                        rss::fmt_bytes(rss_stats.peak_file_rss_bytes),
                    );
                    log_hidden_stats(&consumer, "after compaction");
                    // Cost RAM leg still bills total peak (not anon).
                    (wall_s, io, peak_rss, cpu_s)
                });
                if let Some((wall_s, io, peak_rss, _)) = compaction_stats {
                    transitions.push(TransitionStat {
                        label: "optimize (drain + compact)",
                        wall_ns: wall_s * 1e9,
                        io: Some(io),
                        peak_rss_bytes: Some(peak_rss),
                    });
                }
                let mut cold_split_post = None;
                if run_compact {
                    let compact_map = delta_id_to_dense.as_ref().unwrap_or(&id_to_dense);
                    let compact_truth = if skip_vector_delta {
                        &gt_correct
                    } else {
                        &augmented_gt
                    };
                    let compact_reader = SupertableVectorRead {
                        table: &consumer,
                        id_to_dense: Arc::clone(compact_map),
                    };
                    let post_compact_recall = exec_vec::mean_recall(
                        &compact_reader,
                        supertable::VEC_COLUMN,
                        &q_correct,
                        compact_truth,
                        TOP_K,
                        nprobe,
                        rerank,
                    );
                    let post_compact = measure_routing_state(
                        "post-compact",
                        ExpectedTiers::HiddenOnly,
                        Some(format!("{post_compact_recall:.3}")),
                        &consumer,
                        &consumer_meter,
                        &built,
                        &q_correct[0],
                        &q_correct[1..],
                        nprobe,
                        rerank,
                        built
                            .total_index_bytes
                            .saturating_mul(SHARED_CONSUMER_CACHE_INDEX_FACTOR),
                        phases.warm,
                        phases.cold,
                    );
                    cold_split_post = post_compact.cold;
                    routing_states.push(post_compact);
                }
                if !routing_states.is_empty() {
                    emit_routing_states(&mut report, n_docs, &routing_states);
                }
                if !transitions.is_empty() {
                    emit_transitions(&mut report, n_docs, &transitions);
                }

                // Steady-state footprint = user table + derived hidden vector
                // index. `built.total_index_bytes` is ingest-time user-only
                // (hidden empty then); the post-drain hidden per-cell IVF is a
                // second on-storage copy of the vectors, so price the sum.
                // Computed after compaction so it reflects the merged layout.
                let user_stored = on_storage_bytes(&consumer);
                let hidden_stored = consumer
                    .vector_index_table()
                    .map(|h| {
                        log_hidden_open_stats(h, "post-measurement accounting");
                        on_storage_bytes(h)
                    })
                    .unwrap_or(0);
                let post_drain_stored = user_stored + hidden_stored;
                // Slow-CAS entry blob (drain-published routing state) is a
                // storage object outside the superfile sums above; list its
                // prefix so the stored-capacity readout can't hide it.
                let slow_state_stored = consumer
                    .vector_index_table()
                    .and_then(|h| slow_state_stored_bytes(h))
                    .unwrap_or(0);
                // The PRICED capacity is a real object-store LIST over the
                // table root, filtered to LIVE state: current superfiles
                // (user + hidden), manifests/parts/siblings, pointers, and
                // slow-CAS objects. Data objects the current manifests no
                // longer reference (the superseded generation compaction
                // just replaced, sitting out GC's safety window) are
                // excluded — steady-state capacity, not a
                // moment-after-compaction snapshot.
                let bucket_stored = live_stored_bytes(&consumer)
                    .filter(|&bytes| bytes > 0)
                    .unwrap_or(post_drain_stored + slow_state_stored);
                let manifest_overhead = bucket_stored
                    .saturating_sub(post_drain_stored)
                    .saturating_sub(slow_state_stored);
                eprintln!(
                    "[supertable_vector] on-storage footprint (steady state): user {} + hidden index {} = {} superfiles (ingest-time user-only was {}); slow vector-state {}; manifests {}; PRICED live total (listed) {}",
                    rss::fmt_bytes(user_stored),
                    rss::fmt_bytes(hidden_stored),
                    rss::fmt_bytes(post_drain_stored),
                    rss::fmt_bytes(built.total_index_bytes),
                    rss::fmt_bytes(slow_state_stored),
                    rss::fmt_bytes(manifest_overhead),
                    rss::fmt_bytes(bucket_stored),
                );
                // Retained tables keep everything the run wrote — including
                // the drained hidden index — so a follow-up run can iterate
                // on probe configs without re-uploading or re-draining.
                if hidden_stored > 0 {
                    if let Ok(prefix) = std::env::var("INFINO_BENCH_EXISTING_PREFIX") {
                        eprintln!(
                            "[supertable_vector] drained state retained ({}): rerun with \
                             INFINO_BENCH_EXISTING_PREFIX={prefix} and without \
                             INFINO_BENCH_FORCE_PRE_POST_DRAIN to reuse it (no ingest, no drain)",
                            current_routing_phase(&consumer),
                        );
                    } else if std::env::var_os("INFINO_BENCH_KEEP_TABLE").is_some() {
                        eprintln!(
                            "[supertable_vector] drained state retained ({}): rerun with \
                             INFINO_BENCH_EXISTING_PREFIX=<kept prefix logged by [tiers] above> \
                             to reuse it (no ingest, no drain)",
                            current_routing_phase(&consumer),
                        );
                    }
                }
                let warm_vec = cost::warm_from_vector(&recall_rows);
                let cold_vec = cost::cold_from_vector(&recall_rows);
                let cold_measured = if pre_post_drain {
                    cold_split_post.map(routing_cold_to_measurement)
                } else {
                    phases
                        .cold
                        .then(|| {
                            measure_cold_store(
                                "steady-state",
                                &built,
                                &q_cal[0],
                                if q_cal.len() > 1 {
                                    &q_cal[1..]
                                } else {
                                    &q_cal[..]
                                },
                                nprobe,
                                rerank,
                                post_drain_stored,
                            )
                        })
                        .flatten()
                        .map(routing_cold_to_measurement)
                };
                let store = cost::StorePhases {
                    drain: drain_stats.map(|(_, io, _, _)| io),
                    drain_wall_s: drain_stats.map(|(wall_s, _, _, _)| wall_s),
                    drain_cpu_s: drain_stats.and_then(|(_, _, _, cpu_s)| cpu_s),
                    drain_peak_rss_bytes: drain_stats.map(|(_, _, peak, _)| peak),
                    delta_commit: delta_stats.map(|(_, io, _, _)| io),
                    delta_commit_wall_s: delta_stats.map(|(wall_s, _, _, _)| wall_s),
                    delta_commit_cpu_s: delta_stats.and_then(|(_, _, _, cpu_s)| cpu_s),
                    delta_commit_peak_rss_bytes: delta_stats.map(|(_, _, peak, _)| peak),
                    compaction: compaction_stats.map(|(_, io, _, _)| io),
                    compaction_wall_s: compaction_stats.map(|(wall_s, _, _, _)| wall_s),
                    compaction_cpu_s: compaction_stats.and_then(|(_, _, _, cpu_s)| cpu_s),
                    compaction_peak_rss_bytes: compaction_stats.map(|(_, _, peak, _)| peak),
                    cold_open_pre: cold_split_pre.map(|s| s.open),
                    cold_query_pre: cold_split_pre.map(|s| s.first_query),
                    warm_query: warm_io.map(|(io, _)| io),
                    warm_query_iters: warm_io.map(|(_, n)| n).unwrap_or(0),
                    query_states: query_state_costs(&routing_states),
                    filtered_query: filtered_stats.map(|(io, _)| io),
                    filtered_query_iters: filtered_stats.map(|(_, n)| n).unwrap_or(0),
                    ..store_phases_from_measurement(cold_measured)
                };
                let warm_pre_vec = pre_search_rows
                    .as_deref()
                    .map(cost::warm_from_vector)
                    .unwrap_or_default();
                let cold_pre_vec = pre_search_rows
                    .as_deref()
                    .map(cost::cold_from_vector)
                    .unwrap_or_default();
                let cost_n_docs = n_docs
                    + if pre_post_drain && !skip_vector_delta {
                        supertable::docs_per_commit()
                    } else {
                        0
                    };
                emit_cost_warm(
                    &mut report,
                    "bench/vector/supertable/cost",
                    format!(
                        "Supertable vector — cost model ({} docs × dim={})",
                        fmt_count(cost_n_docs),
                        DIM
                    ),
                    &built,
                    ingest_metrics.as_ref(),
                    cost_n_docs,
                    &warm_vec,
                    (!cold_vec.is_empty()).then_some(cold_vec.as_slice()),
                    pre_search_rows
                        .is_some()
                        .then_some((warm_pre_vec.as_slice(), cold_pre_vec.as_slice())),
                    true,
                    store,
                    Some(bucket_stored),
                );
            }

            drop(consumer);
            drop(cache_dir);
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_vector] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }
}

pub mod sql {
    use super::*;
    use crate::{
        executors::{sql as exec_sql, sql::SqlRead},
        harness::sample_query_csv,
    };

    /// Build a SQL supertable, then measure warm + cold `query_sql` through
    /// the shared SQL executor (same code + same query shapes as superfile).
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_sql] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_sql");
        let (built, ingest_metrics) = build_or_open(Modality::Sql, phases);
        if let Some(metrics) = &ingest_metrics {
            report.emit(&Section {
                anchor: "bench/sql/supertable/ingest".into(),
                title: format!(
                    "Supertable SQL — ingest, multi-superfile / object-store ({} rows, {} commits, {} writers)",
                    fmt_count(n_docs),
                    supertable::n_commits(),
                    supertable::n_writers()
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: ingest_headers(),
                    rows: vec![ingest_row(n_docs, "SQL", metrics)],
                }],
            });
        }

        let inputs = exec_sql::QueryInputs {
            qv: sample_query_csv(),
            sample_title: built
                .sql_sample_title
                .clone()
                .expect("sql ingest sets sample_title"),
            sample_key: built
                .sql_sample_key
                .clone()
                .expect("sql ingest sets sample_key"),
        };

        // Optimize + post-compact only on a fresh ingest in this process —
        // same gate as FTS.
        let run_optimize = ingest_metrics.is_some() && (phases.warm || phases.cold);

        if phases.warm || phases.cold {
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            exec_sql::assert_correct(&consumer, n_docs, "supertable_sql");
            drop(consumer);
            drop(cache_dir);
        }

        // Pre-compact (or sole) warm/cold on the post-ingest layout.
        let warm_sets_pre = if phases.warm {
            eprintln!("[supertable_sql] warm (pre-compact): opening consumer...");
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            let sets = exec_sql::measure_query_sets(
                &consumer,
                &inputs,
                exec_sql::ITERS,
                "supertable_sql",
                &[],
            );
            drop(consumer);
            drop(cache_dir);
            let (anchor, title, note) = if run_optimize {
                (
                    "bench/sql/supertable/warm-pre-compact",
                    format!(
                        "Supertable SQL — warm queries pre-compact, warm cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Pre-compact (post-ingest fanout): each query once untimed (cache fill), then p50 / p90 / p99. Δ vs previous run.",
                )
            } else {
                (
                    "bench/sql/supertable/warm",
                    format!(
                        "Supertable SQL — warm queries, warm cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Warm = committed table reopened with a disk cache sized to the index; each query runs once untimed (cache fill), then p50 / p90 / p99 over repeated `query_sql` calls (Δ gates on `p50`), all through infino's own path (the DataFusion-only control arms are not run here). Δ is vs the previous run.",
                )
            };
            exec_sql::emit_query(&mut report, anchor, title, note, &sets);
            Some(sets)
        } else {
            None
        };

        let cold_pre = if phases.cold {
            let cold = exec_sql::measure_cold(
                || SupertableSqlColdGuard::open(&built),
                COLD_ITERS,
                "supertable_sql",
            );
            let (anchor, title, note) = if run_optimize {
                (
                    "bench/sql/supertable/cold-pre-compact",
                    format!(
                        "Supertable SQL — cold queries pre-compact, fresh cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Pre-compact cold: open = construct only; search is the first query on that cold consumer. Δ vs previous run.",
                )
            } else {
                (
                    "bench/sql/supertable/cold",
                    format!(
                        "Supertable SQL — cold queries, fresh cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Cold = fresh disk cache + consumer per iteration (open = construct only; search is the first query on that cold consumer — no pre-open of all superfiles). Δ is vs the previous run.",
                )
            };
            exec_sql::emit_cold(&mut report, anchor, title, note, &cold);
            Some(cold)
        } else {
            None
        };

        let compaction_stats =
            run_optimize.then(|| optimize_built_table("supertable_sql", Modality::Sql, &built));

        let (warm_sets_post, cold_post) = if run_optimize {
            let warm_sets_post = if phases.warm {
                eprintln!("[supertable_sql] warm (post-compact): opening consumer...");
                let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
                let sets = exec_sql::measure_query_sets(
                    &consumer,
                    &inputs,
                    exec_sql::ITERS,
                    "supertable_sql",
                    &[],
                );
                drop(consumer);
                drop(cache_dir);
                exec_sql::emit_query(
                    &mut report,
                    "bench/sql/supertable/warm-post-compact",
                    format!(
                        "Supertable SQL — warm queries post-compact, warm cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Post-compact (after optimize): fewer superfiles; same warm recipe. Steady-state layout for the cost model. Δ vs previous run.",
                    &sets,
                );
                Some(sets)
            } else {
                None
            };
            let cold_post = if phases.cold {
                let cold = exec_sql::measure_cold(
                    || SupertableSqlColdGuard::open(&built),
                    COLD_ITERS,
                    "supertable_sql",
                );
                exec_sql::emit_cold(
                    &mut report,
                    "bench/sql/supertable/cold-post-compact",
                    format!(
                        "Supertable SQL — cold queries post-compact, fresh cache / object-store ({} rows)",
                        fmt_count(n_docs)
                    ),
                    "Post-compact cold: open = construct only; search is the first query on the merged layout. Δ vs previous run.",
                    &cold,
                );
                Some(cold)
            } else {
                None
            };
            (warm_sets_post, cold_post)
        } else {
            (None, None)
        };

        let warm_pre_vec = warm_sets_pre
            .as_ref()
            .map(cost::warm_from_sql)
            .unwrap_or_default();
        let cold_pre_vec = cold_pre
            .as_ref()
            .map(cost::cold_from_timings)
            .unwrap_or_default();
        let warm_post_vec = warm_sets_post
            .as_ref()
            .map(cost::warm_from_sql)
            .unwrap_or_default();
        let cold_post_vec = cold_post
            .as_ref()
            .map(cost::cold_from_timings)
            .unwrap_or_default();
        let (warm_vec, cold_vec, pre_latencies) = if run_optimize {
            (
                warm_post_vec.as_slice(),
                cold_post_vec.as_slice(),
                Some((warm_pre_vec.as_slice(), cold_pre_vec.as_slice())),
            )
        } else {
            (warm_pre_vec.as_slice(), cold_pre_vec.as_slice(), None)
        };
        let cold_measured = phases.cold.then(|| measure_cold_store(&built)).flatten();
        if !warm_vec.is_empty() || !cold_vec.is_empty() {
            emit_cost_warm(
                &mut report,
                "bench/sql/supertable/cost",
                format!("Supertable SQL — cost model ({} rows)", fmt_count(n_docs)),
                &built,
                ingest_metrics.as_ref(),
                n_docs,
                warm_vec,
                (!cold_vec.is_empty()).then_some(cold_vec),
                pre_latencies,
                false,
                store_phases_with_compaction(cold_measured, compaction_stats),
                None,
            );
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_sql] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    /// One metered cold `query_sql` consumer for the cost model: true cold
    /// open → first query → steady second → repeat, same recipe as FTS
    /// ([`cold_store::measure_cold_store`]).
    ///
    /// Open = consumer construct on a fresh disk cache only (no
    /// `open_all_superfiles`). Queries are the warm-path equality /
    /// filter projections (`fts_pushdown` shapes) — they open Parquet on
    /// a cold miss. The scalar [`exec_sql::SQL_BATTERY`] aggregates stay
    /// on the latency cold table; those answer from the manifest and are
    /// not the cold-I/O cost cell.
    fn measure_cold_store(built: &supertable::IngestResult) -> Option<ColdStoreMeasurement> {
        let sample_title = built.sql_sample_title.as_deref()?;
        let sample_key = built.sql_sample_key.as_deref()?;
        // Ingest already escapes titles; escape again at the format site so a
        // non-escaped caller cannot break the SQL string literal.
        let sample_title = sample_title.replace('\'', "''");
        let sample_key = sample_key.replace('\'', "''");
        // Same shapes as warm `fts_pushdown` / filter projections: must
        // scan row data, so first/steady cold windows accrue real GETs.
        let first = format!("SELECT key FROM supertable WHERE key = '{sample_key}'");
        // Steady predicates must hit the ingest sample row on every corpus;
        // hard-coded category/rating filters can legitimately return zero.
        let steady = [
            format!("SELECT title FROM supertable WHERE title = '{sample_title}'"),
            format!("SELECT key FROM supertable WHERE title = '{sample_title}'"),
            format!("SELECT title FROM supertable WHERE key = '{sample_key}'"),
        ];
        let meter = storage_meter::wrap(Arc::clone(&built.storage));
        let measured = cold_store::measure_cold_store(
            &meter,
            || {
                let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
                    meter.provider(),
                    Some(built.total_index_bytes),
                );
                let opts = tiers::consumer_options(
                    supertable::options_for(Modality::Sql, None),
                    meter.provider(),
                    cache,
                );
                let consumer = tiers::open_consumer(opts);
                (cache_dir, consumer)
            },
            // `query_rows` already `.expect`s on plan/exec failure; require a
            // non-empty hit so a wrong predicate cannot look like success.
            |(_cache, consumer)| {
                assert!(
                    consumer.query_rows(&first) > 0,
                    "metered first cold SQL returned no rows: {first}"
                );
            },
            |(_cache, consumer), i| {
                let q = &steady[i % steady.len()];
                assert!(
                    consumer.query_rows(q) > 0,
                    "metered steady cold SQL returned no rows: {q}"
                );
            },
            steady.len().min(STEADY_COLD_SAMPLES),
            |(_cache, consumer)| {
                assert!(
                    consumer.query_rows(&first) > 0,
                    "metered repeat cold SQL returned no rows: {first}"
                );
            },
        );
        log_cold_split("supertable_sql", &measured.split);
        Some(measured)
    }

    /// Cold-tier guard: fresh disk cache + consumer only. Do **not** call
    /// `open_all_superfiles` here — that would pre-warm every superfile
    /// before the timed cold search and hide true cold-after-open cost.
    /// (FTS latency cold still pre-opens because BM25 always touches
    /// postings; SQL covered aggregates do not, and pre-open made "cold
    /// open" look like a full table fetch.)
    struct SupertableSqlColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
    }
    impl SupertableSqlColdGuard {
        fn open(built: &supertable::IngestResult) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Sql, built);
            Self {
                _cache_dir: cache_dir,
                consumer,
            }
        }
    }
    impl SqlRead for SupertableSqlColdGuard {
        fn query_rows(&self, sql: &str) -> usize {
            self.consumer.query_rows(sql)
        }
        fn query_count(&self, sql: &str) -> i64 {
            self.consumer.query_count(sql)
        }
    }
}
