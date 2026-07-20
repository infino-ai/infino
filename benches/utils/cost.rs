// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cost model for the bench — turns measured latency, footprint, and
//! object-store request counts into dollars, per the rule "a resource
//! costs money only to the extent that holding it blocks the next
//! tenant."
//!
//! Four blocks, kept separate:
//!
//!   1. **Rate card** — the headline dollars, every figure in one of two
//!      units: **$/1M docs** (write path; storage over the stated
//!      retention) and **$/1M queries** (serving; per-query costs are
//!      sub-cent, so a per-query dollar figure would round to $0 and hide
//!      the real number). RAM appears as an instance-sizing fact, not a
//!      dollar line.
//!   2. **Object-store I/O ledger** — measured HEAD/GET/PUT counts and
//!      byte volumes per lifecycle phase, with per-unit normalization
//!      (PUT/commit, GET/query). Counts come from the
//!      [`crate::storage_meter`] wrapper; phases that did not run metered
//!      are omitted, never guessed.
//!   3. **Compute ledger** — one-time phases (ingest/drain/compaction)
//!      and per-query phases priced from measured on-CPU seconds (never a
//!      wall-clock approximation). One-time phases in absolute dollars,
//!      per-query phases per 1M queries.
//!   4. **Serving** — latency per dollar; cold rows include request cost.
//!
//! Local NVMe (file-backed disk-cache mmap) is treated as free.

use std::{collections::HashMap, sync::OnceLock};

use crate::{
    executors::{ColdTiming, fts::FtsQueryStat, sql::QuerySets, vector::RecallRow},
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::fmt_bytes,
    storage_meter::ObjectStoreMeter,
};

/// S3 Standard capacity, USD per GB-month (decimal GB).
const USD_PER_GB_MONTH: f64 = 0.023;
/// USD per PUT request ($5 per 1M).
const USD_PER_PUT: f64 = 5.0e-6;
/// USD per GET or HEAD request ($0.40 per 1M).
const USD_PER_GET: f64 = 4.0e-7;

/// Default assumed retention when turning stored bytes into GB-months.
const DEFAULT_STORAGE_MONTHS: f64 = 1.0;

/// Bytes per GiB (RAM is reasoned about in GiB).
const BYTES_PER_GIB: f64 = (1u64 << 30) as f64;
/// Bytes per GB (object storage is priced per decimal GB).
const BYTES_PER_GB: f64 = 1.0e9;
/// Seconds per hour.
const SECS_PER_HOUR: f64 = 3600.0;
/// Queries per "per-million" pricing unit.
const PER_MILLION: f64 = 1.0e6;
/// Queries per month assumed by the monthly read line. The write line uses
/// the cell's own corpus size (`n_docs`/month) so the summary prices writing
/// THIS table, not a synthetic volume.
const SUMMARY_QUERIES_PER_MONTH: f64 = 1.0e6;
/// Warm fraction of the blended monthly read line (the rest pay the cold
/// per-query cost).
const SUMMARY_READ_WARM_FRACTION: f64 = 0.95;
/// Maintenance cadence assumed by the monthly summary: one drain pass per
/// this many commits.
const SUMMARY_COMMITS_PER_DRAIN: f64 = 16.0;
/// Maintenance cadence assumed by the monthly summary: one compaction
/// (optimize) pass per this many drains.
const SUMMARY_DRAINS_PER_COMPACTION: f64 = 16.0;
/// Padding on the per-query RAM-hold window: a query holds the resident set
/// a little longer than its own p50 (dispatch, response write, scheduler
/// slack between overlapped queries), so the hold is billed at fudge × p50.
/// Residency is otherwise billed strictly per query served — never as a
/// standing calendar-hours line. Bench cost model only: real customer
/// metering must record the exact measured hold time and put any padding
/// in the PRICE, never in the reported quantity.
const QUERY_RAM_HOLD_FUDGE: f64 = 2.0;

/// The instance the model prices against. Default is a portable cloud SKU
/// with local NVMe; override via `INFINO_BENCH_COST_*` env vars.
#[derive(Clone, Debug)]
pub struct Instance {
    pub name: String,
    pub vcpu: u32,
    pub ram_gib: f64,
    pub nvme_gb: f64,
    pub usd_per_hour: f64,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            name: "c7gd.2xlarge".into(),
            vcpu: 8,
            ram_gib: 16.0,
            nvme_gb: 237.0,
            usd_per_hour: 0.3629,
        }
    }
}

impl Instance {
    pub fn current() -> &'static Instance {
        static INSTANCE: OnceLock<Instance> = OnceLock::new();
        INSTANCE.get_or_init(Instance::from_env)
    }

    fn from_env() -> Self {
        let d = Instance::default();
        let s = |k: &str, v: String| std::env::var(k).unwrap_or(v);
        let f = |k: &str, v: f64| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(v)
        };
        let u = |k: &str, v: u32| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(v)
        };
        Instance {
            name: s("INFINO_BENCH_COST_INSTANCE", d.name),
            vcpu: u("INFINO_BENCH_COST_VCPU", d.vcpu),
            ram_gib: f("INFINO_BENCH_COST_RAM_GIB", d.ram_gib),
            nvme_gb: f("INFINO_BENCH_COST_NVME_GB", d.nvme_gb),
            usd_per_hour: f("INFINO_BENCH_COST_USD_PER_HOUR", d.usd_per_hour),
        }
    }

    fn usd_per_sec(&self) -> f64 {
        self.usd_per_hour / SECS_PER_HOUR
    }

    /// Dollar rate of one vCPU-second on this instance.
    fn usd_per_vcpu_sec(&self) -> f64 {
        self.usd_per_sec() / f64::from(self.vcpu.max(1))
    }

    /// Fraction of the instance's RAM a resident set occupies.
    fn ram_share(&self, resident_bytes: u64) -> f64 {
        resident_bytes as f64 / BYTES_PER_GIB / self.ram_gib
    }

    /// RAM-hold leg for a one-time phase, in aggregate vCPU-seconds: `wall ×
    /// peak-RSS share × vcpu`. Expressed in the same aggregate-vCPU-second unit
    /// as measured CPU so `phase_vcpu_seconds` / `compute_usd` price CPU- and
    /// RAM-bound phases uniformly — `compute_usd` divides the `vcpu` back out,
    /// so a RAM-bound phase still bills exactly RSS-share × wall.
    fn ram_leg(&self, wall_s: f64, peak_rss_bytes: Option<u64>) -> f64 {
        wall_s
            * peak_rss_bytes.map(|b| self.ram_share(b)).unwrap_or(0.0)
            * f64::from(self.vcpu.max(1))
    }

    /// Binding aggregate vCPU·s for a one-time phase from MEASURED on-CPU
    /// seconds: `max(measured CPU, RAM-hold leg)`. CPU is never approximated
    /// from wall time — schedstat is the only compute basis; a phase without a
    /// measurement is reported NOT METERED by the caller (never a wall guess).
    fn phase_vcpu_seconds(&self, cpu_s: f64, wall_s: f64, peak_rss_bytes: Option<u64>) -> f64 {
        cpu_s.max(self.ram_leg(wall_s, peak_rss_bytes))
    }

    /// Dollars for measured on-CPU work: aggregate vCPU-seconds (summed across
    /// cores via schedstat) priced at the per-vCPU rate, never the whole-
    /// instance rate. Every measured-CPU row — one-time phases, table open, and
    /// per-query compute — prices through here.
    fn compute_usd(&self, vcpu_s: f64) -> f64 {
        vcpu_s * self.usd_per_vcpu_sec()
    }

    /// RAM-hold leg for one query, in aggregate vCPU-seconds: the resident
    /// set's share of the box (pinned heap + page-cache working set — the
    /// bytes that must be resident for the query to run warm) held for the
    /// query's COMPUTE window (`window × RSS-share × vcpu`), padded by
    /// [`QUERY_RAM_HOLD_FUDGE`]. For a warm query the window is its own p50;
    /// for a cold query it is the same-config warm p50 — once bytes are local
    /// the scoring path holds the set for about the warm window, while the
    /// rest of the cold p50 is off-CPU I/O wait that holds no extra RAM.
    /// This leg is the ONLY place residency is billed: memory cost scales
    /// with queries actually served, never with calendar hours (idle
    /// processes are reaped; keep-warm policy is the operator's line item).
    fn query_ram_leg(&self, window_s: f64, resident_bytes: u64) -> f64 {
        window_s
            * QUERY_RAM_HOLD_FUDGE
            * self.ram_share(resident_bytes)
            * f64::from(self.vcpu.max(1))
    }

    /// Aggregate vCPU·s a query bills: `max(measured on-CPU, RAM-hold leg)` —
    /// the binding resource over its compute window.
    fn per_query_vcpu_seconds(&self, cpu_s: f64, window_s: f64, resident_bytes: u64) -> f64 {
        cpu_s.max(self.query_ram_leg(window_s, resident_bytes))
    }

    /// Per-query dollars from the binding leg (see `per_query_vcpu_seconds`),
    /// priced per-vCPU.
    fn per_query_usd(&self, cpu_s: f64, window_s: f64, resident_bytes: u64) -> f64 {
        self.compute_usd(self.per_query_vcpu_seconds(cpu_s, window_s, resident_bytes))
    }
}

/// Cold open + search latency for one query shape.
pub struct ColdQuery {
    pub name: String,
    pub open_s: f64,
    pub search_s: f64,
    /// Measured on-CPU seconds for the table-open window, when sampled.
    pub open_cpu_s: Option<f64>,
    /// Measured on-CPU seconds for the first-search window, when sampled.
    /// Includes fetch-path on-CPU work (decompress, CRC, cache write) plus
    /// scoring; excludes I/O wait. Priced separately — never copied from warm.
    pub search_cpu_s: Option<f64>,
}

/// Warm query timing and measured on-CPU seconds.
pub type WarmQueryCost = (String, f64, Option<f64>);

/// One named query-residency state's warm/cold object-store windows.
#[derive(Default, Clone, Copy)]
pub struct QueryStateIo {
    pub label: Option<&'static str>,
    pub cold_open: Option<ObjectStoreMeter>,
    pub cold_query: Option<ObjectStoreMeter>,
    /// A second, distinct query on the same cold consumer: the steady cold
    /// per-query fetch once the first query's one-time metadata warmup
    /// (admit-window centroids, Sq8 meta, stable-id blocks) is resident.
    pub cold_second: Option<ObjectStoreMeter>,
    pub cold_repeat: Option<ObjectStoreMeter>,
    pub warm: Option<ObjectStoreMeter>,
    pub warm_iters: u64,
}

/// I/O plus CPU/wall timings for one named query-residency state.
#[derive(Default, Clone, Copy)]
pub struct QueryStateCost {
    pub io: QueryStateIo,
    pub warm_p50_s: Option<f64>,
    pub warm_cpu_s: Option<f64>,
    pub ram_bytes: Option<u64>,
    /// Engine-only settled anon after the state's warm battery: what a
    /// serving process actually pins (consumer handle + state the engine
    /// retains across queries), with freed query scratch purged and bench
    /// harness heap subtracted out.
    pub ram_anon_bytes: Option<u64>,
    /// Settled file-backed resident bytes at the same sample: the mmap
    /// page-cache working set — reclaimable, NVMe-backed, held only while
    /// actively serving warm.
    pub ram_file_settled_bytes: Option<u64>,
    pub cold_open_s: Option<f64>,
    pub cold_open_cpu_s: Option<f64>,
    pub cold_query_s: Option<f64>,
    pub cold_query_cpu_s: Option<f64>,
    /// Wall/CPU of the steady cold query — the per-query cost once the
    /// first query's metadata warmup is resident. Median across the
    /// distinct steady-cold samples (a single draw is the max of a
    /// concurrent GET fan and one object-store straggler can triple it).
    pub cold_second_s: Option<f64>,
    pub cold_second_cpu_s: Option<f64>,
}

impl QueryStateCost {
    /// Resident bytes a query in this state holds to be served: engine-only
    /// pinned heap + settled page-cache working set when both were sampled
    /// (harness overhead excluded), else the state's total RSS, else the
    /// caller's fallback. This is the byte basis of the per-query RAM-hold
    /// leg — "whatever must occupy RAM for the duration of the query".
    fn serving_resident_bytes(&self, fallback: u64) -> u64 {
        match (self.ram_anon_bytes, self.ram_file_settled_bytes) {
            (Some(anon), Some(file)) => anon + file,
            _ => self.ram_bytes.unwrap_or(fallback),
        }
    }

    /// Display form of the serving resident set, split by layer when both
    /// halves were sampled: pinned heap is supertable state (manifest,
    /// summaries, routing slabs) and page cache is superfile data (postings,
    /// centroid regions, rerank payloads). Falls back to the single total.
    fn serving_ram_label(&self, fallback: u64) -> String {
        match (self.ram_anon_bytes, self.ram_file_settled_bytes) {
            (Some(anon), Some(file)) => format!(
                "{} manifest-pinned + {} superfile cache",
                fmt_bytes(anon),
                fmt_bytes(file)
            ),
            _ => fmt_bytes(self.serving_resident_bytes(fallback)),
        }
    }
}

/// Metered object-store I/O for the lifecycle phases of one bench cell.
/// Every field is optional: a phase that wasn't metered is reported as
/// such — the model never substitutes an estimate for a measurement.
#[derive(Default, Clone, Copy)]
pub struct StorePhases {
    /// The ingest window (all commits): superfile uploads (multipart
    /// parts included), manifest parts/lists, pointer CAS writes.
    pub ingest: Option<ObjectStoreMeter>,
    /// The hidden vector-index drain: reads user vector blobs, writes
    /// per-cell superfiles + routing/manifest updates.
    pub drain: Option<ObjectStoreMeter>,
    /// Wall-clock seconds of the drain window, when it ran.
    pub drain_wall_s: Option<f64>,
    /// Measured on-CPU seconds (all-thread schedstat delta) over the drain
    /// window. `Some` ⇒ price compute from this instead of `wall × share`;
    /// `None` ⇒ fall back to the wall-clock model.
    pub drain_cpu_s: Option<f64>,
    /// Peak RSS sampled over the drain window — the drain is billed at
    /// `max(pool CPU share, peak-RSS share)` for its wall duration.
    pub drain_peak_rss_bytes: Option<u64>,
    /// Diagnostic undrained commit inserted between post-drain and
    /// post-delta query states.
    pub delta_commit: Option<ObjectStoreMeter>,
    pub delta_commit_wall_s: Option<f64>,
    pub delta_commit_cpu_s: Option<f64>,
    pub delta_commit_peak_rss_bytes: Option<u64>,
    /// Maintenance compaction (`optimize()`: user + hidden tables) —
    /// reads the small superfiles, writes merged replacements.
    pub compaction: Option<ObjectStoreMeter>,
    /// Wall-clock seconds of the compaction window, when it ran.
    pub compaction_wall_s: Option<f64>,
    /// Measured on-CPU seconds over the compaction window (same semantics as
    /// [`Self::drain_cpu_s`]).
    pub compaction_cpu_s: Option<f64>,
    /// Peak RSS sampled over the compaction window (same billing rule).
    pub compaction_peak_rss_bytes: Option<u64>,
    /// One cold table open on a fresh cache (manifest + pointer + open
    /// blobs) — one-time, amortized across queries on a supertable.
    pub cold_open: Option<ObjectStoreMeter>,
    /// The first query on the cold cache. Under the v1 open discipline
    /// this includes the one-time metadata warmup (admit-window centroid
    /// regions, Sq8 meta, stable-id blocks) alongside the probe — a
    /// once-per-consumer cost, not the steady cold rate.
    pub cold_query: Option<ObjectStoreMeter>,
    /// A second, distinct query on the same cold consumer — the steady
    /// cold per-query fetch once the first query's metadata warmup is
    /// resident. This is the "GETs per query" number for cold traffic.
    pub cold_second_query: Option<ObjectStoreMeter>,
    /// Pre-drain counterparts of `cold_open` / `cold_query`: the transient
    /// shape a fresh table serves (hidden IVF still in INCOMING) until
    /// maintenance drains it. Priced so the cost of querying *before*
    /// maintenance catches up is visible next to the steady state.
    pub cold_open_pre: Option<ObjectStoreMeter>,
    pub cold_query_pre: Option<ObjectStoreMeter>,
    /// The same query repeated on the same *fresh* consumer. Probes
    /// cache fill lag: if the disk cache absorbed the first query this
    /// is ~0 GETs; a repeat of the full fan means foreground reads are
    /// not retained (or background fill has not landed yet).
    pub cold_repeat_query: Option<ObjectStoreMeter>,
    /// Steady-state warm window: [`Self::warm_query_iters`] queries on
    /// the shared, cache-hot consumer — the same consumer the warm
    /// latency battery timed, so I/O and CPU describe the same path.
    pub warm_query: Option<ObjectStoreMeter>,
    pub warm_query_iters: u64,
    /// Explicit lifecycle query states. When populated, the I/O ledger renders
    /// these rows instead of the legacy pre/steady pair above.
    pub query_states: [QueryStateCost; 4],
    /// Filtered-search window ([`Self::filtered_query_iters`] queries)
    /// on the same shared consumer — filtered vs unfiltered GET/query.
    pub filtered_query: Option<ObjectStoreMeter>,
    pub filtered_query_iters: u64,
}

/// Everything one cell (one tier × modality) needs to be priced.
pub struct CellCost<'a> {
    pub ingest_wall_s: f64,
    pub writers: u32,
    /// Peak RSS during the ingest window, when sampled. Ingest is billed
    /// on the *binding* resource — `max(writer-pool CPU share, peak-RSS
    /// share of RAM)` — same rule queries use; `None` bills CPU share.
    pub ingest_peak_rss_bytes: Option<u64>,
    /// Measured on-CPU seconds over the ingest window. `Some` ⇒ price the
    /// CPU leg from this instead of `wall × pool-share`; `None` ⇒ wall model.
    pub ingest_cpu_s: Option<f64>,
    /// Commits in the ingest window, for PUT-per-commit normalization.
    pub n_commits: u64,
    /// Exact PUT count for write paths that are known without metering
    /// (the superfile tier's single `put_atomic`). `None` + no metered
    /// ingest ⇒ the write-request line reports "not metered".
    pub unmetered_put_count: Option<u64>,
    pub stored_bytes: u64,
    pub corpus_bytes: u64,
    pub n_docs: usize,
    pub resident_anon_bytes: u64,
    /// Steady-state (post-drain, on a vector cell) warm latency battery.
    pub warm: &'a [WarmQueryCost],
    /// Cold latency rows (open and search timed separately), steady state.
    pub cold: Option<&'a [ColdQuery]>,
    /// Pre-drain warm battery — the transient shape before maintenance.
    pub warm_pre: Option<&'a [WarmQueryCost]>,
    /// Pre-drain cold latency rows.
    pub cold_pre: Option<&'a [ColdQuery]>,
    /// Measured object-store I/O per phase.
    pub store: StorePhases,
    /// Whether this cell has the vector maintenance lifecycle (drain,
    /// compaction, filtered search, pre/post-drain split). Those ledger
    /// rows always render on such a cell — as "NOT METERED" when the
    /// harness failed to measure them — and never render elsewhere
    /// (an FTS cell has no drain to meter).
    pub vector_cell: bool,
    /// Assumed retention for the capacity line (GB-months). Default 1 month.
    pub storage_months: Option<f64>,
    /// Whether a cold `open` is a one-time table/namespace open that is
    /// amortized across every query (supertable: manifest load + consumer
    /// setup, paid once), rather than per-query latency. For a single
    /// superfile the open is part of each cold read, so this is `false`.
    pub cold_open_amortized: bool,
}

/// `$X` with adaptive precision: two decimals at or above one cent,
/// otherwise two significant digits — sub-cent values never collapse to
/// a meaningless "$0.0000".
fn usd(v: f64) -> String {
    if v == 0.0 {
        return "$0".into();
    }
    if v >= 0.01 {
        return format!("${v:.2}");
    }
    let decimals = ((-v.log10()).ceil() as usize + 1).min(9);
    format!("${v:.decimals$}")
}

/// Per-query dollars expressed at the meaningful scale: `$X/1M`.
fn usd_per_million(per_unit: f64) -> String {
    format!("{}/1M", usd(per_unit * PER_MILLION))
}

/// Per-query cost with both scales visible — prevents comparing $/open to $/1M.
fn usd_per_query_both_scales(per_query: f64) -> String {
    format!("{}/query ({})", usd(per_query), usd_per_million(per_query))
}

/// Latency per dollar: seconds of query latency per dollar of per-query
/// cost (`p50 ÷ $/query`). Not delta-tracked — cost and latency pull it in
/// opposite directions, so neither day-over-day direction is "better".
fn latency_secs_per_usd(per_query_usd: f64, latency_s: f64) -> f64 {
    latency_s / per_query_usd.max(f64::MIN_POSITIVE)
}

/// `s/$` cell rendered at count scale (`11.7K`).
fn latency_per_usd_cell(per_query_usd: f64, latency_s: f64) -> Cell {
    text(fmt_count(
        latency_secs_per_usd(per_query_usd, latency_s) as usize
    ))
}

/// Event count for the maintenance cadence line: integers plain, fractional
/// cadences with two decimals (`1` / `0.06`).
fn fmt_events(n: f64) -> String {
    if (n - n.round()).abs() < 1e-9 {
        format!("{n:.0}")
    } else {
        format!("{n:.2}")
    }
}

fn usd_per_gb(v: f64) -> String {
    // Three decimals below $0.10: the S3 capacity rate is $0.023/GB-mo and
    // a two-decimal "$0.02/GB" would misstate the rate the math applies.
    if v < 0.1 {
        format!("${v:.3}/GB")
    } else {
        format!("${v:.2}/GB")
    }
}

fn storage_months() -> f64 {
    static MONTHS: OnceLock<f64> = OnceLock::new();
    *MONTHS.get_or_init(|| {
        std::env::var("INFINO_BENCH_COST_STORAGE_MONTHS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(DEFAULT_STORAGE_MONTHS)
    })
}

fn fmt_vcpu_seconds(s: f64) -> String {
    if s >= 10.0 {
        format!("{s:.1}")
    } else if s >= 0.01 {
        format!("{s:.2}")
    } else if s > 0.0 {
        // Sub-centi vCPU·s: show enough digits that vCPU·s × per-vCPU-rate
        // visibly reconciles with the $ column (0.00068 must not read "0.00").
        let decimals = ((-s.log10()).ceil() as usize + 1).min(6);
        format!("{s:.decimals$}")
    } else {
        "0.00".into()
    }
}

fn fmt_wall_seconds(s: f64) -> String {
    if s >= 10.0 {
        format!("{s:.1} s")
    } else {
        format!("{s:.2} s")
    }
}

/// Request dollars for one metered window: PUT + LIST at the PUT/list rate,
/// HEAD + GET at the GET rate. DELETE is free on S3, so it is counted but not
/// priced.
fn request_usd(io: &ObjectStoreMeter) -> f64 {
    (io.put_count + io.list_count) as f64 * USD_PER_PUT + io.read_requests() as f64 * USD_PER_GET
}

/// "N PUT + M GET (+ K HEAD / LIST / DELETE)" — the request-count cell of an I/O row.
fn fmt_requests(io: &ObjectStoreMeter) -> String {
    let mut parts = Vec::new();
    if io.put_count > 0 {
        parts.push(format!("{} PUT", io.put_count));
    }
    if io.get_count > 0 {
        parts.push(format!("{} GET", io.get_count));
    }
    if io.head_count > 0 {
        parts.push(format!("{} HEAD", io.head_count));
    }
    if io.list_count > 0 {
        parts.push(format!("{} LIST", io.list_count));
    }
    if io.delete_count > 0 {
        parts.push(format!("{} DELETE", io.delete_count));
    }
    if parts.is_empty() {
        "0".into()
    } else {
        parts.join(" + ")
    }
}

fn fmt_uploaded(io: &ObjectStoreMeter) -> String {
    if io.put_bytes == 0 {
        "—".into()
    } else {
        fmt_bytes(io.put_bytes)
    }
}

fn fmt_downloaded(io: &ObjectStoreMeter) -> String {
    if io.get_bytes == 0 {
        "—".into()
    } else {
        fmt_bytes(io.get_bytes)
    }
}

pub fn emit(report: &mut Report, anchor: &str, title: String, c: &CellCost) {
    let inst = Instance::current();
    let retention_months = c.storage_months.unwrap_or_else(storage_months);

    // ---- Write path: ingest + drain + compaction (compute and requests).
    // Each phase is billed at its binding share — pool CPU or peak-RSS
    // share of RAM, whichever is larger — for its full wall duration.
    // Compute is priced ONLY from measured on-CPU seconds (schedstat, I/O
    // wait excluded). A phase that ran but whose CPU wasn't sampled is
    // reported NOT METERED — never back-filled with a wall-clock guess.
    let ingest_compute = c
        .ingest_cpu_s
        .map(|cpu| inst.phase_vcpu_seconds(cpu, c.ingest_wall_s, c.ingest_peak_rss_bytes))
        .map(|vcpu| inst.compute_usd(vcpu));
    let drain_wall_s = c.store.drain_wall_s.unwrap_or(0.0);
    let drain_compute = c
        .store
        .drain_cpu_s
        .map(|cpu| inst.phase_vcpu_seconds(cpu, drain_wall_s, c.store.drain_peak_rss_bytes))
        .map(|vcpu| inst.compute_usd(vcpu));
    let delta_wall_s = c.store.delta_commit_wall_s.unwrap_or(0.0);
    let delta_compute = c
        .store
        .delta_commit_cpu_s
        .map(|cpu| inst.phase_vcpu_seconds(cpu, delta_wall_s, c.store.delta_commit_peak_rss_bytes))
        .map(|vcpu| inst.compute_usd(vcpu));
    let compaction_wall_s = c.store.compaction_wall_s.unwrap_or(0.0);
    let compaction_compute = c
        .store
        .compaction_cpu_s
        .map(|cpu| {
            inst.phase_vcpu_seconds(cpu, compaction_wall_s, c.store.compaction_peak_rss_bytes)
        })
        .map(|vcpu| inst.compute_usd(vcpu));

    let ingest_req_usd = match (c.store.ingest, c.unmetered_put_count) {
        (Some(io), _) => request_usd(&io),
        (None, Some(puts)) => puts as f64 * USD_PER_PUT,
        (None, None) => 0.0,
    };
    let drain_req_usd = c.store.drain.map(|io| request_usd(&io)).unwrap_or(0.0);
    let delta_req_usd = c
        .store
        .delta_commit
        .map(|io| request_usd(&io))
        .unwrap_or(0.0);
    let compaction_req_usd = c.store.compaction.map(|io| request_usd(&io)).unwrap_or(0.0);

    let write_compute = ingest_compute.unwrap_or(0.0)
        + drain_compute.unwrap_or(0.0)
        + delta_compute.unwrap_or(0.0)
        + compaction_compute.unwrap_or(0.0);
    let write_requests = ingest_req_usd + drain_req_usd + delta_req_usd + compaction_req_usd;
    let write_total = write_compute + write_requests;
    let write_per_million_docs = if c.n_docs > 0 {
        write_total / (c.n_docs as f64 / PER_MILLION)
    } else {
        0.0
    };
    // "$X per 1M docs" for a one-time maintenance phase's requests.
    let per_million_docs = |usd_total: f64| {
        if c.n_docs > 0 {
            usd_total / (c.n_docs as f64 / PER_MILLION)
        } else {
            0.0
        }
    };

    // ---- Storage capacity ----
    let stored_gb = c.stored_bytes as f64 / BYTES_PER_GB;
    let gb_months = stored_gb * retention_months;
    let storage_month = gb_months * USD_PER_GB_MONTH;

    // ---- Warm query battery (priced from MEASURED on-CPU seconds) ----
    // Only entries with a sampled cpu are priced; an unmetered warm query is
    // omitted from the battery rather than back-filled with a wall guess.
    let warm_costs: Vec<(f64, f64, String)> = c
        .warm
        .iter()
        .filter_map(|(name, p50_s, cpu_s)| {
            cpu_s.map(|cpu| {
                let per_q = inst.per_query_usd(cpu, *p50_s, c.resident_anon_bytes);
                (per_q, *p50_s, name.clone())
            })
        })
        .collect();
    let (min_q_cost, max_q_cost, fastest_name, fastest_p50) = if warm_costs.is_empty() {
        (0.0, 0.0, String::new(), 0.0)
    } else {
        warm_costs.iter().fold(
            (f64::INFINITY, 0.0_f64, String::new(), f64::INFINITY),
            |(min_c, max_c, fast_name, fast_p50), (cost, p50, name)| {
                let (fast_name, fast_p50) = if *p50 < fast_p50 {
                    (name.clone(), *p50)
                } else {
                    (fast_name, fast_p50)
                };
                (min_c.min(*cost), max_c.max(*cost), fast_name, fast_p50)
            },
        )
    };

    // Anchor cold row: the shape whose open/search latency and metered I/O
    // represent "one cold query" in the rate card and ledgers.
    let anchor_cold = c.cold.and_then(|rows| {
        rows.iter()
            .find(|q| q.name == "ten_term_or")
            .or_else(|| rows.first())
    });

    // The same-config warm p50 — the cold query's RAM-hold window (the heap
    // is held for the compute portion of a cold query, about the warm window;
    // the rest of the cold p50 is off-CPU I/O wait holding no extra heap).
    let warm_window_for = |name: &str| -> Option<f64> {
        c.warm
            .iter()
            .find(|(n, _, _)| n == name)
            .or_else(|| c.warm.first())
            .map(|(_, p50_s, _)| *p50_s)
    };

    // Per-query cold dollars = binding(MEASURED cold-search on-CPU, RAM leg
    // over the warm-scale compute window) + measured object-store requests
    // for the first-query fetch window. Cold search CPU is metered separately
    // (includes decompress/decode/scoring); it is never copied from warm.
    let cold_query_req_usd = c.store.cold_query.map(|io| request_usd(&io));
    let cold_query_usd = anchor_cold.map(|q| {
        let window = warm_window_for(&q.name).unwrap_or(0.0);
        q.search_cpu_s
            .map(|cpu| inst.per_query_usd(cpu, window, c.resident_anon_bytes))
            .unwrap_or(0.0)
            + cold_query_req_usd.unwrap_or(0.0)
    });

    // ---- Block 1: rate card ----
    let warm_query_cell = if warm_costs.is_empty() {
        "—".into()
    } else if (max_q_cost - min_q_cost).abs() < f64::EPSILON {
        format!(
            "{} queries @ {} p50 ({})",
            usd_per_million(min_q_cost),
            fmt_time(fastest_p50 * 1e9),
            fastest_name,
        )
    } else {
        format!(
            "{}–{} queries ({}–{} p50 battery)",
            usd(min_q_cost * PER_MILLION),
            usd_per_million(max_q_cost),
            fmt_time(fastest_p50 * 1e9),
            fmt_time(
                warm_costs
                    .iter()
                    .map(|(_, p50, _)| *p50)
                    .fold(0.0_f64, f64::max)
                    * 1e9,
            ),
        )
    };

    let has_drain = c.store.drain.is_some() || c.store.drain_wall_s.is_some();
    let has_delta = c.store.delta_commit.is_some() || c.store.delta_commit_wall_s.is_some();
    let has_compaction = c.store.compaction.is_some() || c.store.compaction_wall_s.is_some();
    let write_label = match (has_drain, has_delta, has_compaction) {
        (true, true, true) => "Write path (ingest + drain + delta + optimize)",
        (true, false, true) => "Write path (ingest + drain + optimize)",
        (true, _, false) => "Write path (ingest + hidden-index drain)",
        (false, true, true) => "Write path (ingest + delta + optimize)",
        (false, _, true) => "Write path (ingest + optimize)",
        (false, true, false) => "Write path (ingest + delta)",
        (false, false, false) => "Write path (ingest)",
    };
    let query_states: Vec<&QueryStateCost> = c
        .store
        .query_states
        .iter()
        .filter(|state| state.io.label.is_some())
        .collect();
    let mut rate_rows = vec![
        vec![
            text("Storage"),
            text(format!(
                "{}/1M docs ({} × {retention_months:.0} mo retention)",
                usd(per_million_docs(storage_month)),
                usd_per_gb(USD_PER_GB_MONTH),
            )),
        ],
        vec![
            text(write_label),
            text(format!(
                "{} compute + {} requests → {} total ({}/1M docs)",
                usd(write_compute),
                usd(write_requests),
                usd(write_total),
                usd(write_per_million_docs),
            )),
        ],
    ];
    if query_states.is_empty() {
        rate_rows.push(vec![
            text("Warm query (marginal, binding resource)"),
            text(warm_query_cell),
        ]);
    }

    if query_states.is_empty()
        && let Some(q) = anchor_cold
    {
        if let Some(per_q) = cold_query_usd.filter(|_| cold_query_req_usd.is_some()) {
            let io = c.store.cold_query.expect("guarded by cold_query_req_usd");
            rate_rows.push(vec![
                text("Cold query (CPU + requests)"),
                text(format!(
                    "{} queries — {} GET/query, {}/query fetched ({} search, {})",
                    usd_per_million(per_q),
                    io.get_count,
                    fmt_bytes(io.get_bytes),
                    fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        } else {
            rate_rows.push(vec![
                text("Cold query (latency only — requests not metered)"),
                text(format!(
                    "{} open + {} search ({})",
                    fmt_time(q.open_s * 1e9),
                    fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        }
        if c.cold_open_amortized {
            let open_io = c
                .store
                .cold_open
                .map(|io| {
                    format!(
                        " · {} GET, {} fetched",
                        io.read_requests(),
                        fmt_bytes(io.get_bytes)
                    )
                })
                .unwrap_or_default();
            rate_rows.push(vec![
                text("Table open (one-time, amortized)"),
                text(format!(
                    "{}{open_io} — manifest + consumer, paid once per open",
                    fmt_time(q.open_s * 1e9),
                )),
            ]);
        }
    }

    let rate_card = Block {
        subtitle: format!(
            "Rate card — {} docs, {} stored",
            fmt_count(c.n_docs),
            fmt_bytes(c.stored_bytes),
        ),
        headers: vec!["Line".into(), "Infino (measured)".into()],
        rows: rate_rows,
    };

    // ---- Block 2: object-store I/O ledger ----
    let mut io_rows: Vec<Vec<Cell>> = Vec::new();
    // A lifecycle phase this cell *has* but the harness failed to measure
    // renders as a loud placeholder — a phase must never silently vanish.
    let not_metered_row = |label: &str| -> Vec<Cell> {
        vec![
            text(label),
            text("NOT METERED"),
            text("—"),
            text("—"),
            text("—"),
            text("—"),
        ]
    };
    match (c.store.ingest, c.unmetered_put_count) {
        (Some(io), _) => {
            io_rows.push(vec![
                text(format!("Ingest ({} commits)", c.n_commits)),
                text(fmt_requests(&io)),
                text(fmt_uploaded(&io)),
                text(fmt_downloaded(&io)),
                text(format!(
                    "{}/1M docs",
                    usd(per_million_docs(request_usd(&io)))
                )),
                metric(request_usd(&io), usd(request_usd(&io)), Better::Lower),
            ]);
        }
        (None, Some(puts)) => {
            let req = puts as f64 * USD_PER_PUT;
            io_rows.push(vec![
                text(format!("Ingest ({} commits)", c.n_commits)),
                text(format!("{puts} PUT (exact, unmetered)")),
                text(fmt_bytes(c.stored_bytes)),
                text("—"),
                text(format!("{}/1M docs", usd(per_million_docs(req)))),
                metric(req, usd(req), Better::Lower),
            ]);
        }
        (None, None) => io_rows.push(not_metered_row("Ingest (opened pre-built)")),
    }
    let one_time_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>, per_unit: &str| {
            match io {
                Some(io) => {
                    let per_unit = if per_unit.is_empty() {
                        format!("{}/1M docs", usd(per_million_docs(request_usd(&io))))
                    } else {
                        per_unit.to_string()
                    };
                    rows.push(vec![
                        text(label),
                        text(fmt_requests(&io)),
                        text(fmt_uploaded(&io)),
                        text(fmt_downloaded(&io)),
                        text(per_unit),
                        metric(request_usd(&io), usd(request_usd(&io)), Better::Lower),
                    ]);
                }
                None if c.vector_cell => rows.push(not_metered_row(label)),
                None => {}
            }
        };
    let per_query_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>| match io {
            Some(io) => {
                let per_million = request_usd(&io) * PER_MILLION;
                rows.push(vec![
                    text(label),
                    text(fmt_requests(&io)),
                    text(fmt_uploaded(&io)),
                    text(fmt_downloaded(&io)),
                    metric(
                        io.get_count as f64,
                        format!("{}/query", io.get_count),
                        Better::Lower,
                    ),
                    metric(
                        per_million,
                        format!("{}/1M queries", usd(per_million)),
                        Better::Lower,
                    ),
                ]);
            }
            None if c.vector_cell => rows.push(not_metered_row(label)),
            None => {}
        };
    one_time_row(&mut io_rows, "Drain", c.store.drain, "");
    one_time_row(&mut io_rows, "Delta commit", c.store.delta_commit, "");
    one_time_row(&mut io_rows, "Optimize", c.store.compaction, "");
    // Averaged multi-query windows on the shared cache-hot consumer: the
    // same consumer the warm latency battery timed, so the ledger's warm
    // I/O and the compute ledger's warm CPU describe one path.
    let averaged_row =
        |rows: &mut Vec<Vec<Cell>>, label: &str, io: Option<ObjectStoreMeter>, iters: u64| match io
        {
            Some(io) => {
                let iters = iters.max(1);
                let per_query_get = io.get_count as f64 / iters as f64;
                let per_query_usd = request_usd(&io) / iters as f64;
                let per_million = per_query_usd * PER_MILLION;
                // Enough decimals that e.g. 1 GET / 20 queries reads 0.05,
                // not a doubled-looking "0.1".
                let get_cell = if per_query_get > 0.0 && per_query_get < 0.1 {
                    format!("{per_query_get:.2}/query")
                } else {
                    format!("{per_query_get:.1}/query")
                };
                rows.push(vec![
                    text(label),
                    text(format!("{} / {iters}q", fmt_requests(&io))),
                    text(fmt_uploaded(&io)),
                    text(fmt_downloaded(&io)),
                    metric(per_query_get, get_cell, Better::Lower),
                    metric(
                        per_million,
                        format!("{}/1M queries", usd(per_million)),
                        Better::Lower,
                    ),
                ]);
            }
            None if c.vector_cell => rows.push(not_metered_row(label)),
            None => {}
        };
    if query_states.is_empty() {
        if c.vector_cell {
            one_time_row(
                &mut io_rows,
                "Cold table open (pre-drain)",
                c.store.cold_open_pre,
                "1/open",
            );
            per_query_row(
                &mut io_rows,
                "Cold query (pre-drain, transient)",
                c.store.cold_query_pre,
            );
        }
        one_time_row(&mut io_rows, "Cold table open", c.store.cold_open, "1/open");
        per_query_row(
            &mut io_rows,
            "Cold query (first on cold cache, +metadata warmup)",
            c.store.cold_query,
        );
        per_query_row(
            &mut io_rows,
            "Cold query (second, steady cold)",
            c.store.cold_second_query,
        );
        let fill = match (c.store.cold_query, c.store.cold_repeat_query) {
            (Some(q), Some(r)) => Some(q.merge_background_fill(&r)),
            (Some(q), None) => Some(q.background_fill_meter()),
            (None, Some(r)) => Some(r.background_fill_meter()),
            (None, None) => None,
        };
        per_query_row(&mut io_rows, "Cache fill (during cold query)", fill);
        per_query_row(
            &mut io_rows,
            "Repeat query on cold consumer",
            c.store.cold_repeat_query,
        );
        averaged_row(
            &mut io_rows,
            "Warm query (shared consumer, cache hot)",
            c.store.warm_query,
            c.store.warm_query_iters,
        );
    } else {
        for state in &query_states {
            let label = state.io.label.expect("filtered query state has a label");
            one_time_row(
                &mut io_rows,
                &format!("Open — {label}"),
                state.io.cold_open,
                "1/open",
            );
            per_query_row(
                &mut io_rows,
                &format!("Cold 1st (+metadata warmup) — {label}"),
                state.io.cold_query,
            );
            per_query_row(
                &mut io_rows,
                &format!("Cold 2nd (steady cold) — {label}"),
                state.io.cold_second,
            );
            // Background lazy→mmap fill concurrent with the cold/repeat
            // windows — counted separately so query GETs stay foreground-only.
            let fill = match (state.io.cold_query, state.io.cold_repeat) {
                (Some(q), Some(r)) => Some(q.merge_background_fill(&r)),
                (Some(q), None) => Some(q.background_fill_meter()),
                (None, Some(r)) => Some(r.background_fill_meter()),
                (None, None) => None,
            };
            per_query_row(&mut io_rows, &format!("Fill — {label}"), fill);
            per_query_row(
                &mut io_rows,
                &format!("Repeat — {label}"),
                state.io.cold_repeat,
            );
            averaged_row(
                &mut io_rows,
                &format!("Warm — {label}"),
                state.io.warm,
                state.io.warm_iters,
            );
        }
    }
    averaged_row(
        &mut io_rows,
        "Filtered warm (~10%)",
        c.store.filtered_query,
        c.store.filtered_query_iters,
    );
    let io_ledger = (!io_rows.is_empty()).then(|| Block {
        subtitle: "Object-store I/O — measured requests and transfer bytes.".into(),
        headers: vec![
            "Phase".into(),
            "Requests".into(),
            "Uploaded".into(),
            "Downloaded".into(),
            "Per-unit".into(),
            "Cost".into(),
        ],
        rows: io_rows,
    });

    // ---- Block 3: compute ledger ----
    // One-time-phase row from MEASURED on-CPU seconds. `None` cpu ⇒ NOT
    // METERED (the phase ran but schedstat was unavailable) — never a
    // wall-clock substitute. Shared by ingest / drain / compaction so the
    // Some/None handling and cell layout live in one place.
    let phase_row =
        |label: String, wall_s: f64, peak_rss: Option<u64>, cpu_s: Option<f64>| -> Vec<Cell> {
            let Some(cpu) = cpu_s else {
                return vec![
                    text(label),
                    text(fmt_wall_seconds(wall_s)),
                    text("N/A"),
                    text("N/A"),
                    text("N/A"),
                    text("N/A"),
                ];
            };
            let ram = inst.ram_leg(wall_s, peak_rss);
            let vcpu = inst.phase_vcpu_seconds(cpu, wall_s, peak_rss);
            let usd_v = inst.compute_usd(vcpu);
            let binding = if ram > cpu { "RAM" } else { "CPU" };
            vec![
                text(label),
                text(fmt_wall_seconds(wall_s)),
                text(fmt_vcpu_seconds(cpu)),
                text(peak_rss.map(fmt_bytes).unwrap_or_else(|| "N/A".into())),
                text(binding),
                metric(usd_v, usd(usd_v), Better::Lower),
            ]
        };
    let mut compute_rows = vec![phase_row(
        "Ingest".into(),
        c.ingest_wall_s,
        c.ingest_peak_rss_bytes,
        c.ingest_cpu_s,
    )];
    if c.store.drain_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Drain".to_string(),
            drain_wall_s,
            c.store.drain_peak_rss_bytes,
            c.store.drain_cpu_s,
        ));
    } else if c.vector_cell {
        compute_rows.push(phase_row("Drain".to_string(), 0.0, None, None));
    }
    if c.store.delta_commit_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Delta commit".to_string(),
            delta_wall_s,
            c.store.delta_commit_peak_rss_bytes,
            c.store.delta_commit_cpu_s,
        ));
    }
    if c.store.compaction_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Optimize".to_string(),
            compaction_wall_s,
            c.store.compaction_peak_rss_bytes,
            c.store.compaction_cpu_s,
        ));
    } else if c.vector_cell {
        compute_rows.push(phase_row("Optimize".to_string(), 0.0, None, None));
    }
    if query_states.is_empty()
        && let Some(q) = anchor_cold
    {
        let open_label = format!("Open — {}", q.name);
        // Table open is compute-bound (manifest parse + reader CRC: measured
        // cpu ≈ wall), so it's priced from its MEASURED on-CPU seconds. NOT
        // METERED (never latency × share) when unsampled.
        compute_rows.push(match q.open_cpu_s {
            Some(cpu) => {
                let open_usd = inst.compute_usd(cpu);
                vec![
                    text(open_label),
                    text(fmt_wall_seconds(q.open_s)),
                    text(fmt_vcpu_seconds(cpu)),
                    text(fmt_bytes(c.resident_anon_bytes)),
                    text("CPU"),
                    metric(open_usd, usd(open_usd), Better::Lower),
                ]
            }
            None => vec![
                text(open_label),
                text(fmt_wall_seconds(q.open_s)),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
        // Cold search CPU: MEASURED on-CPU during the search window (decompress,
        // decode, scoring — not copied from warm), with the RAM leg over the
        // warm-scale compute window. NOT METERED when unsampled.
        compute_rows.push(match q.search_cpu_s {
            Some(cpu) => {
                let window = warm_window_for(&q.name).unwrap_or(0.0);
                let ram = inst.query_ram_leg(window, c.resident_anon_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu, window, c.resident_anon_bytes);
                let per_q = inst.compute_usd(vcpu);
                let binding = if ram > cpu { "RAM" } else { "CPU" };
                vec![
                    text(format!("Cold — {}", q.name)),
                    text(fmt_time(q.search_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu)),
                    text(fmt_bytes(c.resident_anon_bytes)),
                    text(binding),
                    metric(
                        per_q * PER_MILLION,
                        usd_per_query_both_scales(per_q),
                        Better::Lower,
                    ),
                ]
            }
            None => vec![
                text(format!("Cold — {}", q.name)),
                text(fmt_time(q.search_s * 1e9)),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
    }
    if query_states.is_empty()
        && let Some((name, p50_s, cpu_s)) = c
            .warm
            .iter()
            .find(|(n, _, _)| n == "ten_term_or")
            .or_else(|| c.warm.first())
    {
        // Warm query priced from MEASURED on-CPU seconds (per-vCPU) with the
        // RAM leg over its own p50 window. NOT METERED when unsampled.
        compute_rows.push(match cpu_s {
            Some(cpu) => {
                let ram = inst.query_ram_leg(*p50_s, c.resident_anon_bytes);
                let vcpu = inst.per_query_vcpu_seconds(*cpu, *p50_s, c.resident_anon_bytes);
                let per_q = inst.compute_usd(vcpu);
                let binding = if ram > *cpu { "RAM" } else { "CPU" };
                vec![
                    text(format!("Warm — {name}")),
                    text(fmt_time(*p50_s * 1e9)),
                    text(fmt_vcpu_seconds(*cpu)),
                    text(fmt_bytes(c.resident_anon_bytes)),
                    text(binding),
                    metric(
                        per_q * PER_MILLION,
                        usd_per_query_both_scales(per_q),
                        Better::Lower,
                    ),
                ]
            }
            None => vec![
                text(format!("Warm — {name}")),
                text(fmt_time(*p50_s * 1e9)),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
    }
    for state in &query_states {
        let label = state.io.label.expect("filtered query state has a label");
        compute_rows.push(match (state.cold_open_s, state.cold_open_cpu_s) {
            (Some(wall_s), Some(cpu_s)) => {
                let ram_bytes = state.ram_bytes.unwrap_or(c.resident_anon_bytes);
                let ram = inst.ram_leg(wall_s, Some(ram_bytes));
                let billed = cpu_s.max(ram);
                let usd_v = inst.compute_usd(billed);
                vec![
                    text(format!("Open — {label}")),
                    text(fmt_wall_seconds(wall_s)),
                    text(fmt_vcpu_seconds(cpu_s)),
                    text(fmt_bytes(ram_bytes)),
                    text(if ram > cpu_s { "RAM" } else { "CPU" }),
                    metric(usd_v, usd(usd_v), Better::Lower),
                ]
            }
            _ => vec![
                text(format!("Open — {label}")),
                text(
                    state
                        .cold_open_s
                        .map(fmt_wall_seconds)
                        .unwrap_or_else(|| "N/A".into()),
                ),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
        compute_rows.push(match (state.cold_query_s, state.cold_query_cpu_s) {
            (Some(wall_s), Some(cpu_s)) => {
                let warm_window = state.warm_p50_s.unwrap_or(0.0);
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let ram = inst.query_ram_leg(warm_window, ram_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes);
                let per_q = inst.compute_usd(vcpu);
                vec![
                    text(format!("Cold 1st (warmup) — {label}")),
                    text(fmt_time(wall_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu_s)),
                    text(state.serving_ram_label(c.resident_anon_bytes)),
                    text(if ram > cpu_s { "RAM" } else { "CPU" }),
                    metric(
                        per_q * PER_MILLION,
                        usd_per_query_both_scales(per_q),
                        Better::Lower,
                    ),
                ]
            }
            _ => vec![
                text(format!("Cold 1st (warmup) — {label}")),
                text(
                    state
                        .cold_query_s
                        .map(|seconds| fmt_time(seconds * 1e9))
                        .unwrap_or_else(|| "N/A".into()),
                ),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
        if let (Some(wall_s), Some(cpu_s)) = (state.cold_second_s, state.cold_second_cpu_s) {
            let warm_window = state.warm_p50_s.unwrap_or(0.0);
            let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
            let ram = inst.query_ram_leg(warm_window, ram_bytes);
            let vcpu = inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes);
            let per_q = inst.compute_usd(vcpu);
            compute_rows.push(vec![
                text(format!("Cold 2nd (steady) — {label}")),
                text(fmt_time(wall_s * 1e9)),
                text(fmt_vcpu_seconds(cpu_s)),
                text(state.serving_ram_label(c.resident_anon_bytes)),
                text(if ram > cpu_s { "RAM" } else { "CPU" }),
                metric(
                    per_q * PER_MILLION,
                    usd_per_query_both_scales(per_q),
                    Better::Lower,
                ),
            ]);
        }
        compute_rows.push(match (state.warm_p50_s, state.warm_cpu_s) {
            (Some(p50_s), Some(cpu_s)) => {
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let ram = inst.query_ram_leg(p50_s, ram_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu_s, p50_s, ram_bytes);
                let per_q = inst.compute_usd(vcpu);
                vec![
                    text(format!("Warm — {label}")),
                    text(fmt_time(p50_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu_s)),
                    text(state.serving_ram_label(c.resident_anon_bytes)),
                    text(if ram > cpu_s { "RAM" } else { "CPU" }),
                    metric(
                        per_q * PER_MILLION,
                        usd_per_query_both_scales(per_q),
                        Better::Lower,
                    ),
                ]
            }
            _ => vec![
                text(format!("Warm — {label}")),
                text(
                    state
                        .warm_p50_s
                        .map(|seconds| fmt_time(seconds * 1e9))
                        .unwrap_or_else(|| "N/A".into()),
                ),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
            ],
        });
    }
    let compute_ledger = Block {
        subtitle: "Compute — actual CPU time and resident RAM; binding determines cost.".into(),
        headers: vec![
            "Phase".into(),
            "Wall / p50".into(),
            "CPU (s)".into(),
            "RAM".into(),
            "Binding".into(),
            "Cost".into(),
        ],
        rows: compute_rows,
    };

    // ---- Block 4: serving ----
    let mut serving_rows: Vec<Vec<Cell>> = Vec::new();
    // Steady-state per-query dollars for the monthly summary: the LAST
    // populated query state (post-compact when the lifecycle ran) is the
    // shape a long-lived table serves.
    let mut steady_warm: Option<(String, f64)> = None;
    let mut steady_cold: Option<(String, f64)> = None;
    if query_states.is_empty() {
        serving_rows.extend(c.warm.iter().filter_map(|(name, p50_s, cpu_s)| {
            let cpu = (*cpu_s)?;
            let per_q = inst.per_query_usd(cpu, *p50_s, c.resident_anon_bytes);
            let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
            Some(vec![
                text(format!("{name} — warm")),
                text(fmt_time(p50_s * 1e9)),
                metric(
                    queries_per_usd,
                    format!("{queries_per_usd:.0}"),
                    Better::Higher,
                ),
                latency_per_usd_cell(per_q, *p50_s),
                text(usd(per_q * PER_MILLION)),
            ])
        }));
        if let Some((name, p50_s, Some(cpu))) = c
            .warm
            .iter()
            .find(|(n, _, _)| n == "ten_term_or")
            .or_else(|| c.warm.first())
        {
            let per_q = inst.per_query_usd(*cpu, *p50_s, c.resident_anon_bytes);
            steady_warm = Some((format!("warm ({name})"), per_q));
        }
        if let (Some(q), Some(per_q)) = (anchor_cold, cold_query_usd) {
            let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
            serving_rows.push(vec![
                text(format!("{} — cold", q.name)),
                text(fmt_time(q.search_s * 1e9)),
                metric(
                    queries_per_usd,
                    format!("{queries_per_usd:.0}"),
                    Better::Higher,
                ),
                latency_per_usd_cell(per_q, q.search_s),
                text(usd(per_q * PER_MILLION)),
            ]);
            steady_cold = Some((format!("cold ({})", q.name), per_q));
        }
    } else {
        for state in &query_states {
            let label = state.io.label.expect("filtered query state has a label");
            if let (Some(p50_s), Some(cpu_s)) = (state.warm_p50_s, state.warm_cpu_s) {
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let per_q = inst.per_query_usd(cpu_s, p50_s, ram_bytes);
                let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
                serving_rows.push(vec![
                    text(format!("warm — {label}")),
                    text(fmt_time(p50_s * 1e9)),
                    metric(
                        queries_per_usd,
                        format!("{queries_per_usd:.0}"),
                        Better::Higher,
                    ),
                    latency_per_usd_cell(per_q, p50_s),
                    text(usd(per_q * PER_MILLION)),
                ]);
                steady_warm = Some((format!("warm — {label}"), per_q));
            }
            // First cold query: one-time metadata warmup — a rate reference,
            // never the blended cold leg.
            if let (Some(wall_s), Some(cpu_s), Some(io)) = (
                state.cold_query_s,
                state.cold_query_cpu_s,
                state.io.cold_query,
            ) {
                let warm_window = state.warm_p50_s.unwrap_or(0.0);
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let per_q = inst.per_query_usd(cpu_s, warm_window, ram_bytes) + request_usd(&io);
                let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
                serving_rows.push(vec![
                    text(format!(
                        "cold 1st, one-time warmup — {label} ({} GET)",
                        io.get_count
                    )),
                    text(fmt_time(wall_s * 1e9)),
                    metric(
                        queries_per_usd,
                        format!("{queries_per_usd:.0}"),
                        Better::Higher,
                    ),
                    latency_per_usd_cell(per_q, wall_s),
                    text(usd(per_q * PER_MILLION)),
                ]);
                // Fallback steady leg when no second-query window was metered.
                if state.io.cold_second.is_none() {
                    steady_cold = Some((format!("cold — {label} ({} GET)", io.get_count), per_q));
                }
            }
            // Second (steady) cold query: the per-query price cold traffic
            // actually pays — this is the blended cold leg.
            if let (Some(wall_s), Some(cpu_s), Some(io)) = (
                state.cold_second_s,
                state.cold_second_cpu_s,
                state.io.cold_second,
            ) {
                let warm_window = state.warm_p50_s.unwrap_or(0.0);
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let per_q = inst.per_query_usd(cpu_s, warm_window, ram_bytes) + request_usd(&io);
                let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
                serving_rows.push(vec![
                    text(format!("cold steady — {label} ({} GET)", io.get_count)),
                    text(fmt_time(wall_s * 1e9)),
                    metric(
                        queries_per_usd,
                        format!("{queries_per_usd:.0}"),
                        Better::Higher,
                    ),
                    latency_per_usd_cell(per_q, wall_s),
                    text(usd(per_q * PER_MILLION)),
                ]);
                steady_cold = Some((
                    format!("cold steady — {label} ({} GET)", io.get_count),
                    per_q,
                ));
            }
        }
    }
    let serving = Block {
        subtitle: "Serving — query latency and cost by lifecycle state; s/$ is \
                   latency per dollar (p50 seconds ÷ $/query)."
            .into(),
        headers: vec![
            "Query".into(),
            "p50".into(),
            "queries/$".into(),
            "s/$".into(),
            "$/1M queries".into(),
        ],
        rows: serving_rows,
    };

    // ---- Block 5: monthly cost summary ----
    // The standing bill for one table at the assumed steady load. Residency
    // is NOT a standing line: the resident set a query needs in order to be
    // served — pinned heap (manifest, routing state) plus the page-cache
    // working set — is billed inside each query's price through the RAM-hold
    // leg (`query_ram_leg`: resident share × fudged query window), and its
    // per-layer bytes are shown on the compute ledger's query rows. Memory
    // cost therefore scales with queries actually served, never with
    // calendar hours; idle processes are reaped, and any keep-warm-while-
    // idle policy is the operator's line item. All inputs are measured — a
    // line without a measurement is omitted, never guessed.
    let mut summary_rows: Vec<Vec<Cell>> = vec![vec![
        text("Storage"),
        text(format!(
            "{} stored for {} docs, {retention_months:.0} mo retention",
            fmt_bytes(c.stored_bytes),
            fmt_count(c.n_docs),
        )),
        metric(storage_month, usd(storage_month), Better::Lower),
    ]];
    // Warm and cold read lines are rate references (empty $/month); the
    // blended line — most queries warm, a small tail paying the cold fetch —
    // is the billed monthly read cost. Each per-query price already carries
    // its RAM-hold leg for the full resident set.
    let blended_read_q = match (&steady_warm, &steady_cold) {
        (Some((_, warm_q)), Some((_, cold_q))) => {
            Some(warm_q * SUMMARY_READ_WARM_FRACTION + cold_q * (1.0 - SUMMARY_READ_WARM_FRACTION))
        }
        (Some((_, warm_q)), None) => Some(*warm_q),
        _ => None,
    };
    if let Some((label, per_q)) = &steady_warm {
        summary_rows.push(vec![
            text(format!(
                "Reads — {} queries/mo, {label}",
                fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
            )),
            text(usd_per_million(*per_q)),
            text(""),
        ]);
    }
    if let Some((label, per_q)) = &steady_cold {
        summary_rows.push(vec![
            text(format!(
                "Reads — {} queries/mo, {label}",
                fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
            )),
            text(usd_per_million(*per_q)),
            text(""),
        ]);
    }
    if steady_warm.is_some()
        && steady_cold.is_some()
        && let Some(blended_q) = blended_read_q
    {
        let month = blended_q * SUMMARY_QUERIES_PER_MONTH;
        summary_rows.push(vec![
            text(format!(
                "Reads — {} queries/mo, {:.0}% warm / {:.0}% cold blend",
                fmt_count(SUMMARY_QUERIES_PER_MONTH as usize),
                SUMMARY_READ_WARM_FRACTION * 100.0,
                (1.0 - SUMMARY_READ_WARM_FRACTION) * 100.0,
            )),
            text(usd_per_million(blended_q)),
            metric(month, usd(month), Better::Lower),
        ]);
    }
    // Writes priced at the corpus scale the bench actually measured — the
    // whole table written once per month — covering COMMITS only (ingest +
    // the delta commit). Drain and compaction move to the maintenance lines
    // below so the total never double-counts them.
    let writes_month = ingest_compute.unwrap_or(0.0)
        + ingest_req_usd
        + delta_compute.unwrap_or(0.0)
        + delta_req_usd;
    summary_rows.push(vec![
        text(format!(
            "Writes — {} docs/mo (commits)",
            fmt_count(c.n_docs)
        )),
        text(format!("{}/1M docs", usd(per_million_docs(writes_month)))),
        metric(writes_month, usd(writes_month), Better::Lower),
    ]);
    // Maintenance: per-event rates for open / drain / compaction, then one
    // billed line at the stated cadence — a drain pass every
    // `SUMMARY_COMMITS_PER_DRAIN` commits, a compaction pass every
    // `SUMMARY_DRAINS_PER_COMPACTION` drains, one table open per pass.
    let drain_pass_usd = has_drain.then(|| drain_compute.unwrap_or(0.0) + drain_req_usd);
    let compact_pass_usd =
        has_compaction.then(|| compaction_compute.unwrap_or(0.0) + compaction_req_usd);
    let steady_open_usd = query_states
        .last()
        .map(|state| {
            let compute = match (state.cold_open_s, state.cold_open_cpu_s) {
                (Some(wall_s), Some(cpu_s)) => {
                    let ram_bytes = state.ram_bytes.unwrap_or(c.resident_anon_bytes);
                    inst.compute_usd(cpu_s.max(inst.ram_leg(wall_s, Some(ram_bytes))))
                }
                _ => 0.0,
            };
            let req = state.io.cold_open.map(|io| request_usd(&io)).unwrap_or(0.0);
            compute + req
        })
        .or_else(|| {
            anchor_cold.map(|q| {
                q.open_cpu_s.map(|cpu| inst.compute_usd(cpu)).unwrap_or(0.0)
                    + c.store.cold_open.map(|io| request_usd(&io)).unwrap_or(0.0)
            })
        });
    if let Some(open) = steady_open_usd {
        summary_rows.push(vec![
            text("Open — cold table open"),
            text(format!("{}/open", usd(open))),
            text(""),
        ]);
    }
    if let Some(drain) = drain_pass_usd {
        summary_rows.push(vec![
            text(format!(
                "Drain — one pass over {} commits",
                fmt_events(SUMMARY_COMMITS_PER_DRAIN)
            )),
            text(format!("{}/pass", usd(drain))),
            text(""),
        ]);
    }
    if let Some(compact) = compact_pass_usd {
        summary_rows.push(vec![
            text("Compaction — one optimize pass"),
            text(format!("{}/pass", usd(compact))),
            text(""),
        ]);
    }
    let maintenance_month = drain_pass_usd.map(|drain| {
        let drains_mo = c.n_commits.max(1) as f64 / SUMMARY_COMMITS_PER_DRAIN;
        let compacts_mo = drains_mo / SUMMARY_DRAINS_PER_COMPACTION;
        let opens_mo = drains_mo + compacts_mo;
        let month = drains_mo * drain
            + compacts_mo * compact_pass_usd.unwrap_or(0.0)
            + opens_mo * steady_open_usd.unwrap_or(0.0);
        summary_rows.push(vec![
            text(format!(
                "Maintenance — drain / {} commits, compaction / {} drains",
                fmt_events(SUMMARY_COMMITS_PER_DRAIN),
                fmt_events(SUMMARY_DRAINS_PER_COMPACTION),
            )),
            text(format!(
                "{} drains + {} compactions + {} opens/mo",
                fmt_events(drains_mo),
                fmt_events(compacts_mo),
                fmt_events(opens_mo),
            )),
            metric(month, usd(month), Better::Lower),
        ]);
        month
    });
    let monthly_total = storage_month
        + blended_read_q
            .map(|q| q * SUMMARY_QUERIES_PER_MONTH)
            .unwrap_or(0.0)
        + writes_month
        + maintenance_month.unwrap_or(0.0);
    summary_rows.push(vec![
        text("Total (storage + blended reads + writes + maintenance)"),
        text("—"),
        metric(monthly_total, usd(monthly_total), Better::Lower),
    ]);
    let monthly_summary = Block {
        subtitle: format!(
            "Monthly cost summary — one open table, {} queries served + {} docs \
             written per month, steady state.",
            fmt_count(SUMMARY_QUERIES_PER_MONTH as usize),
            fmt_count(c.n_docs),
        ),
        headers: vec!["Line".into(), "Basis".into(), "$/month".into()],
        rows: summary_rows,
    };

    let mut blocks = vec![rate_card];
    if let Some(io_ledger) = io_ledger {
        blocks.push(io_ledger);
    }
    blocks.push(compute_ledger);
    blocks.push(serving);
    blocks.push(monthly_summary);

    report.emit(&Section {
        anchor: anchor.into(),
        title,
        note:
            "Measured values only; N/A means the phase was not sampled. Δ is vs the previous run."
                .into(),
        blocks,
    });
}

/// Flatten cold `(open, search)` timings keyed by query name into cost
/// rows. Shared by the FTS and SQL runners (both measure per-query
/// `ColdTiming` maps).
pub fn cold_from_timings(cold: &HashMap<&'static str, ColdTiming>) -> Vec<ColdQuery> {
    cold.iter()
        .map(|(name, t)| ColdQuery {
            name: (*name).to_string(),
            open_s: t.open.as_secs_f64(),
            search_s: t.search.as_secs_f64(),
            open_cpu_s: t.open_cpu_s,
            search_cpu_s: t.search_cpu_s,
        })
        .collect()
}

/// Flatten warm FTS stats into `(name, p50_seconds, measured_cpu_seconds)`.
pub fn warm_from_fts(stats: &[FtsQueryStat]) -> Vec<WarmQueryCost> {
    stats
        .iter()
        .map(|s| (s.name.to_string(), s.warm.p50.as_secs_f64(), s.cpu_s))
        .collect()
}

/// Flatten warm SQL query sets into `(name, p50_seconds, measured_cpu_seconds)`.
pub fn warm_from_sql(sets: &QuerySets) -> Vec<WarmQueryCost> {
    sets.scalar
        .iter()
        .chain(&sets.tvf)
        .chain(&sets.fts_pushdown)
        .chain(&sets.agg_idx)
        .map(|s| (s.name.to_string(), s.warm.p50.as_secs_f64(), s.cpu_s))
        .collect()
}

/// Flatten warm vector recall rows into `(label, p50_seconds, measured_cpu_seconds)`.
pub fn warm_from_vector(rows: &[RecallRow]) -> Vec<WarmQueryCost> {
    rows.iter()
        .filter_map(|r| {
            r.warm.as_ref().map(|w| {
                let label = if r.params.is_empty() || r.params == "—" {
                    r.target.clone()
                } else {
                    format!("{} ({})", r.target, r.params)
                };
                (label, w.warm.p50.as_secs_f64(), w.cpu_s)
            })
        })
        .collect()
}

/// Flatten cold vector recall rows into `(label, open, search)` for the cost model.
pub fn cold_from_vector(rows: &[RecallRow]) -> Vec<ColdQuery> {
    rows.iter()
        .filter_map(|r| {
            r.cold.map(|t| {
                let label = if r.params.is_empty() || r.params == "—" {
                    r.target.clone()
                } else {
                    format!("{} ({})", r.target, r.params)
                };
                ColdQuery {
                    name: label,
                    open_s: t.open.as_secs_f64(),
                    search_s: t.search.as_secs_f64(),
                    open_cpu_s: t.open_cpu_s,
                    search_cpu_s: t.search_cpu_s,
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_instance() -> Instance {
        Instance {
            name: "test".into(),
            vcpu: 8,
            ram_gib: 16.0,
            nvme_gb: 237.0,
            usd_per_hour: 0.3629,
        }
    }

    #[test]
    fn phase_bills_measured_cpu_when_it_exceeds_ram() {
        let inst = test_instance();
        // Measured on-CPU is billed verbatim (I/O wait already excluded) when
        // it exceeds the RAM-hold leg — no wall-clock substitution.
        assert!((inst.phase_vcpu_seconds(5.0, 10.0, None) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ram_bound_phase_bills_rss_share_for_full_wall() {
        let inst = test_instance();
        // 8 GiB peak on 16 GiB = 50% RAM share over a 10s wall on an 8-vCPU
        // box = 0.5 × 10 × 8 = 40 aggregate vCPU·s of RAM hold; a smaller
        // measured CPU is dominated by it → phase is RAM-bound.
        let eight_gib = 8u64 << 30;
        assert!((inst.phase_vcpu_seconds(1.0, 10.0, Some(eight_gib)) - 40.0).abs() < 1e-9);
        // Measured CPU above the RAM leg binds on CPU and is billed as-is.
        assert!((inst.phase_vcpu_seconds(41.0, 10.0, Some(eight_gib)) - 41.0).abs() < 1e-9);
        // A RAM-bound phase still bills exactly RSS-share × wall in dollars —
        // compute_usd divides the vcpu back out: 40 vCPU·s ⇒ 0.5 × 10 × $/s.
        let ram_bound_usd = inst.compute_usd(inst.phase_vcpu_seconds(1.0, 10.0, Some(eight_gib)));
        assert!((ram_bound_usd - 0.5 * 10.0 * inst.usd_per_sec()).abs() < 1e-12);
    }

    #[test]
    fn query_cpu_priced_per_vcpu_from_measured_seconds() {
        let inst = test_instance();
        // Tiny resident heap ⇒ RAM leg negligible ⇒ measured compute binds. A
        // query measured at 10× the on-CPU seconds costs 10× more, and it's
        // priced at the PER-VCPU rate (whole-instance rate ÷ vcpu), never the
        // whole-instance rate — the bug that inflated cold queries.
        let small = 1u64 << 20;
        let cheap = inst.per_query_usd(0.001, 0.001, small);
        let dear = inst.per_query_usd(0.010, 0.001, small);
        assert!(dear > cheap);
        assert!((dear / cheap - 10.0).abs() < 1e-6);
        assert!((dear - 0.010 * inst.usd_per_sec() / 8.0).abs() < 1e-12);
    }

    #[test]
    fn fmt_vcpu_seconds_reconciles_with_per_vcpu_rate() {
        // 0.00542 vCPU·s must not display as "0.00" — the user audits by
        // multiplying vCPU·s × rate and comparing to the $ column.
        assert_eq!(fmt_vcpu_seconds(0.00542), "0.0054");
        assert_eq!(fmt_vcpu_seconds(0.000678), "0.00068");
        assert_eq!(fmt_vcpu_seconds(0.05), "0.05");
    }

    #[test]
    fn cold_search_cpu_priced_from_measured_not_warm() {
        let inst = test_instance();
        // 0.05 vCPU·s measured cold search @ per-vCPU rate — NOT copied from warm.
        let cold = inst.compute_usd(0.05);
        assert!(
            (cold * PER_MILLION - 0.63).abs() < 0.05,
            "got ${}/1M",
            cold * PER_MILLION
        );
        // Whole-instance rate would 8× overcharge to ~$5/1M — the old bug.
        assert!(cold * PER_MILLION < 1.0);
    }

    #[test]
    fn usd_never_collapses_sub_cent_values_to_zero() {
        assert_eq!(usd(0.0), "$0");
        assert_eq!(usd(1.014), "$1.01");
        assert_eq!(usd(0.02), "$0.02");
        // Two significant digits below one cent instead of "$0.0000".
        assert_eq!(usd(2.8e-5), "$0.000028");
        assert_eq!(usd(7.0e-5), "$0.000070");
        assert_eq!(usd(0.0028), "$0.0028");
    }

    #[test]
    fn per_million_scales_per_query_dollars() {
        // 175 GET/query at $0.40/1M requests = $70 per 1M queries.
        let per_query = 175.0 * USD_PER_GET;
        assert_eq!(usd_per_million(per_query), "$70.00/1M");
    }

    #[test]
    fn request_usd_prices_puts_lists_and_reads() {
        let io = ObjectStoreMeter {
            head_count: 10,
            get_count: 90,
            get_bytes: 0,
            put_count: 1000,
            put_bytes: 0,
            list_count: 50,
            delete_count: 20,
            ..Default::default()
        };
        // (1000 PUT + 50 LIST) × $5e-6 + 100 reads × $4e-7; DELETE unpriced.
        let expected = 1050.0 * 5.0e-6 + 100.0 * 4.0e-7;
        assert!((request_usd(&io) - expected).abs() < 1e-12);
    }
}
