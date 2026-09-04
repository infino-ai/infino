// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-warm serving-state diagnostic: local NVMe cache fully populated,
//! OS page cache deliberately dropped.
//!
//! The main bench measures two serving states: **warm** (cache files local
//! AND their pages resident in the OS page cache) and **cold** (nothing
//! local — the full object-store fetch). This diagnostic measures the state
//! between them — every byte on local disk, no byte in RAM — which is the
//! latency a query pays after another tenant's traffic evicts this table's
//! page-cache working set. That number decides how far RAM can be
//! oversubscribed on a multi-tenant node: if a disk re-fault is cheap, the
//! per-tenant serving cost sits near the NVMe-occupancy floor; if it is
//! expensive, capacity must be provisioned near the RAM-resident ceiling
//! (see the cost model's provisioned-occupancy block).
//!
//! Dropping the page cache leaves the query code path IDENTICAL to warm —
//! the only variable is where the pages come from. The drop is a
//! three-step pass (see `drop_page_cache`): `sweep_once()` to unmap
//! mmap-promoted cache entries (`fadvise` skips pages a mapping still
//! references), `sync()` so dirty fill pages become droppable, then
//! `posix_fadvise(DONTNEED)` over every file under the cache root plus
//! every regular file the process holds open (centroid spill, manifest
//! parts).
//!
//! Three batteries over the same shared consumer, post-drain:
//! - **RAM-warm**: the ordinary warm battery (baseline).
//! - **disk-warm**: the page cache is dropped before EVERY query, so each
//!   sample faults its working set from local disk.
//! - **re-warmed**: the same battery again with no drops — proves the
//!   working set re-promoted to page cache and recovers the baseline.
//!
//! Validity evidence, printed per battery: object-store GETs (must be 0 —
//! disk-warm, not accidentally cold), physical bytes read from storage
//! (`/proc/self/io` `read_bytes` — near zero when RAM-warm, the working
//! set when disk-warm), and the settled file-backed RSS after the battery.
//!
//! Scale follows `INFINO_BENCH_SUPERTABLE_DOCS`. Invoked as
//! `cargo bench -- disk-warm`.

use std::{
    fs::{self, File},
    hint::black_box,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use infino::{
    VectorSearchOptions,
    supertable::{Supertable, reader_cache::DiskCacheStore},
};
use rustix::fs::{Advice, fadvise, sync};

use crate::{
    corpus::{self, dim},
    diag_common::percentile,
    ingest::supertable::{self as ingest, Modality, VEC_COLUMN},
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, context, metric, text},
    rss,
    storage_meter::{self, MeteredStorage, ObjectStoreMeter},
    tiers,
};

/// Result set size per query — matches the vector cell's battery.
const TOP_K: usize = 10;
/// Queries per battery. Same count as the vector cell's correctness battery,
/// so per-query cell coverage matches the reference warm numbers.
const N_QUERIES: usize = 100;
/// Query-vector generation seed/sigma — mirror the vector cell's
/// correctness battery (`QUERY_CORRECTNESS_SEED` / `QUERY_SIGMA`) so the
/// probes route like the reference battery's.
const QUERY_SEED: u64 = 17;
const QUERY_SIGMA: f32 = 0.05;
/// Iterations of the RAM-warm and re-warmed batteries (p50 over
/// `WARM_ITERS × N_QUERIES` samples). The disk-warm battery runs each query
/// once — every sample is preceded by a full page-cache drop, so iterations
/// are independent by construction.
const WARM_ITERS: usize = 3;
/// Cache budget = this multiple of the built index, mirroring the main
/// bench's shared-consumer sizing (the hidden index roughly doubles a
/// vector table's working set after the drain).
const CACHE_INDEX_FACTOR: u64 = 2;
/// Ceiling on waiting for background cache fills to settle after the fill
/// pass, so "disk-warm" starts from a complete local cache.
const WARM_SETTLE_TIMEOUT: Duration = Duration::from_secs(600);
/// Nanoseconds per second, for report metric cells.
const SEC_TO_NANOS: f64 = 1e9;

/// One battery's samples plus its validity evidence.
struct BatteryResult {
    label: &'static str,
    samples: Vec<Duration>,
    /// Object-store window over the battery — 0 GETs proves disk-warm
    /// rather than accidentally cold.
    io: ObjectStoreMeter,
    /// Physical bytes read from the storage layer (`/proc/self/io`
    /// `read_bytes` delta) — proves disk-warm queries actually hit the
    /// device rather than lingering page cache.
    physical_read_bytes: u64,
    /// Settled file-backed RSS after the battery (page-cache working set).
    file_rss_after: Option<u64>,
}

pub fn run() {
    let n_docs = ingest::n_docs();
    eprintln!(
        "[disk-warm] building vector supertable ({} docs) and draining the hidden index...",
        fmt_count(n_docs)
    );
    let prepared = ingest::prepare_corpus(Modality::Vector);
    let built = ingest::build_on_storage(Modality::Vector, &prepared);
    let queries = {
        let vectors = prepared.vectors().expect("vector corpus");
        let base = &vectors.as_slice()[..n_docs * dim()];
        corpus::bench_queries(base, n_docs, N_QUERIES, QUERY_SEED, true, QUERY_SIGMA)
    };
    drop(prepared);

    let meter = storage_meter::wrap(Arc::clone(&built.storage));
    let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
        meter.provider(),
        Some(built.total_index_bytes.saturating_mul(CACHE_INDEX_FACTOR)),
    );
    // Kept for the page-cache drop: mmap-promoted entries must be
    // `sweep_once()`-advised (PTEs dropped) before `fadvise` can evict
    // their pages — see `drop_page_cache`.
    let cache_for_drop = Arc::clone(&cache);
    let consumer = tiers::open_consumer(tiers::consumer_options(
        ingest::options_for(Modality::Vector, None),
        meter.provider(),
        cache,
    ));
    consumer
        .drain_vectors_to_cells_sync()
        .expect("hidden cell drain");

    // Fill: run the full battery once so every probed cell's blocks land in
    // the local disk cache, settle background fills, then once more so the
    // page cache is hot for the RAM-warm baseline.
    eprintln!("[disk-warm] filling the local disk cache (2 battery passes)...");
    for query in &queries {
        black_box(search(&consumer, query));
    }
    consumer
        .wait_until_warm(WARM_SETTLE_TIMEOUT)
        .expect("cache fills settled");
    for query in &queries {
        black_box(search(&consumer, query));
    }
    rss::log_rss_breakdown("disk-warm: after fill");

    let warm = run_battery("RAM-warm", &consumer, &meter, &queries, WARM_ITERS, None);
    let disk = run_battery(
        "disk-warm",
        &consumer,
        &meter,
        &queries,
        1,
        Some((cache_dir.path(), cache_for_drop.as_ref())),
    );
    let rewarmed = run_battery("re-warmed", &consumer, &meter, &queries, WARM_ITERS, None);

    if disk.io.get_count > 0 {
        eprintln!(
            "[disk-warm] WARNING: {} object-store GET(s) during the disk-warm battery — the \
             local cache did not fully absorb the working set (undersized budget or eviction); \
             the disk-warm latency above includes object-store fetches and overstates the \
             NVMe re-fault cost",
            disk.io.get_count
        );
    }

    let batteries = [warm, disk, rewarmed];
    let mut report = Report::load("disk-warm");
    report.emit(&Section {
        anchor: "bench/vector/supertable/disk-warm".into(),
        title: format!(
            "Disk-warm serving state — supertable vector ({} docs × dim={})",
            fmt_count(n_docs),
            dim()
        ),
        note: "The serving state between warm and cold: local disk cache fully populated, OS \
               page cache dropped (`fadvise(DONTNEED)` before every disk-warm query). Same \
               consumer and code path as the warm battery — only the page residency differs. \
               Valid only when the disk-warm battery shows 0 GETs (else it is partially \
               object-store cold). `phys read` is the /proc/self/io read_bytes delta — the \
               bytes actually faulted from the device. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "State".into(),
                "p50".into(),
                "p90".into(),
                "p99".into(),
                "GET (battery)".into(),
                "phys read".into(),
                "file RSS after".into(),
            ],
            rows: batteries.iter().map(battery_row).collect(),
        }],
    });
    report.save();
}

fn battery_row(battery: &BatteryResult) -> Vec<Cell> {
    let mut samples = battery.samples.clone();
    let p50 = percentile(&mut samples, 50);
    let p90 = percentile(&mut samples, 90);
    let p99 = percentile(&mut samples, 99);
    vec![
        text(battery.label),
        metric(
            p50.as_secs_f64() * SEC_TO_NANOS,
            fmt_time(p50.as_secs_f64() * SEC_TO_NANOS),
            Better::Lower,
        ),
        context(
            p90.as_secs_f64() * SEC_TO_NANOS,
            fmt_time(p90.as_secs_f64() * SEC_TO_NANOS),
            Better::Lower,
        ),
        context(
            p99.as_secs_f64() * SEC_TO_NANOS,
            fmt_time(p99.as_secs_f64() * SEC_TO_NANOS),
            Better::Lower,
        ),
        context(
            battery.io.get_count as f64,
            format!("{}", battery.io.get_count),
            Better::Lower,
        ),
        text(rss::fmt_bytes(battery.physical_read_bytes)),
        text(
            battery
                .file_rss_after
                .map(rss::fmt_bytes)
                .unwrap_or_else(|| "—".into()),
        ),
    ]
}

/// Run one battery. With `drop_state` set (cache root + cache store), the
/// page cache under that root (plus every open regular file) is dropped
/// before EVERY query; the drop itself is excluded from the timed sample.
fn run_battery(
    label: &'static str,
    consumer: &Supertable,
    meter: &MeteredStorage,
    queries: &[Vec<f32>],
    iters: usize,
    drop_state: Option<(&Path, &DiskCacheStore)>,
) -> BatteryResult {
    let io_before = meter.snapshot();
    let physical_before = proc_read_bytes();
    let mut samples = Vec::with_capacity(iters * queries.len());
    let mut dropped_files = 0usize;
    let mut dropped_bytes = 0u64;
    let mut swept_mmaps = 0u64;
    for _ in 0..iters {
        for query in queries {
            if let Some((root, cache)) = drop_state {
                let (files, bytes, swept) = drop_page_cache(root, cache);
                dropped_files = files;
                dropped_bytes = bytes;
                swept_mmaps = swept;
            }
            let t0 = Instant::now();
            black_box(search(consumer, query));
            samples.push(t0.elapsed());
        }
    }
    let io = meter.snapshot().since(&io_before);
    let physical_read_bytes = proc_read_bytes().saturating_sub(physical_before);
    let file_rss_after = rss::settled_rss_breakdown().map(|(_, _, file, _)| file);
    let mut sorted = samples.clone();
    let p50 = percentile(&mut sorted, 50);
    eprintln!(
        "[disk-warm] {label}: p50 {} over {} samples; {} GET / {} down; phys read {}{}",
        fmt_time(p50.as_secs_f64() * SEC_TO_NANOS),
        samples.len(),
        io.get_count,
        rss::fmt_bytes(io.get_bytes),
        rss::fmt_bytes(physical_read_bytes),
        if drop_state.is_some() {
            format!(
                "; per-query drop swept {swept_mmaps} mmap(s), advised {dropped_files} file(s) / {}",
                rss::fmt_bytes(dropped_bytes)
            )
        } else {
            String::new()
        },
    );
    BatteryResult {
        label,
        samples,
        io,
        physical_read_bytes,
        file_rss_after,
    }
}

/// One default-config vector query on the shared consumer — the same call
/// shape the vector cell's warm battery times.
fn search(consumer: &Supertable, query: &[f32]) -> usize {
    let batches = consumer
        .reader()
        .expect("reader")
        .vector_search(
            VEC_COLUMN,
            query,
            TOP_K,
            VectorSearchOptions::default(),
            None,
            None,
        )
        .expect("vector_search");
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Evict the disk cache's pages (and every regular file this process holds
/// open — centroid spill, manifest parts) from the OS page cache. Three
/// steps, each necessary:
/// 1. `sweep_once()` — `MADV_DONTNEED` on every mmap-promoted cache entry
///    (the bench cache config disables the idle threshold, so the sweep
///    covers all of them). `fadvise` skips pages a mapping still
///    references; the sweep drops the PTEs so the pages become plain
///    unmapped page cache. Without this step the whole drop is a measured
///    no-op for mmap'd entries.
/// 2. `sync()` — `fadvise(DONTNEED)` also skips dirty pages, and freshly
///    filled cache blocks may still be in writeback.
/// 3. `fadvise(DONTNEED)` by path under `root` and over every open fd.
///
/// Returns (files, bytes advised under `root`, mmaps swept).
fn drop_page_cache(root: &Path, cache: &DiskCacheStore) -> (usize, u64, u64) {
    let swept = cache.sweep_once();
    sync();
    let mut files = 0usize;
    let mut bytes = 0u64;
    advise_tree(root, &mut files, &mut bytes);
    advise_open_fds();
    (files, bytes, swept)
}

/// `fadvise(DONTNEED)` every regular file under `dir`, recursively.
fn advise_tree(dir: &Path, files: &mut usize, bytes: &mut u64) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            advise_tree(&path, files, bytes);
        } else if meta.is_file()
            && let Ok(file) = File::open(&path)
        {
            let _ = fadvise(&file, 0, None, Advice::DontNeed);
            *files += 1;
            *bytes += meta.len();
        }
    }
}

/// `fadvise(DONTNEED)` every regular file this process holds open, by
/// re-opening its `/proc/self/fd` target — fadvise acts on the inode's page
/// cache, so a fresh fd to the same file evicts the original's pages. This
/// reaches page-cache-backed files that live outside the cache root (the
/// slow-CAS centroid spill in TMPDIR). Non-file targets (pipes, sockets,
/// devices, deleted temp files) resolve to non-file or missing paths and
/// are skipped.
fn advise_open_fds() {
    let Ok(entries) = fs::read_dir("/proc/self/fd") else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        if !target.is_absolute() || target.starts_with("/proc") || target.starts_with("/dev") {
            continue;
        }
        if fs::metadata(&target).map(|m| m.is_file()).unwrap_or(false)
            && let Ok(file) = File::open(&target)
        {
            let _ = fadvise(&file, 0, None, Advice::DontNeed);
        }
    }
}

/// Bytes this process has caused to be fetched from the storage layer
/// (`/proc/self/io` `read_bytes`) — page-cache hits don't count, so the
/// delta over a battery is the physically faulted volume.
fn proc_read_bytes() -> u64 {
    fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("read_bytes:")
                    .and_then(|v| v.trim().parse().ok())
            })
        })
        .unwrap_or(0)
}
