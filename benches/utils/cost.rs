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
    report::{Better, Block, Cell, Report, Section, context, metric, text},
    rss::fmt_bytes,
    storage_meter::{ObjectStoreMeter, background_fill_meter, merge_background_fill},
};

/// S3 Standard capacity, USD per GB-month (decimal GB).
const USD_PER_GB_MONTH: f64 = 0.023;
/// USD per PUT request ($5 per 1M).
const USD_PER_PUT: f64 = 5.0e-6;
/// USD per GET or HEAD request ($0.40 per 1M).
const USD_PER_GET: f64 = 4.0e-7;
/// Default network egress rate, USD per GB out to the client — the AWS S3 /
/// EC2 data-transfer-out-to-internet first tier (~$0.09/GB for the first
/// 10 TB/mo; Azure/GCP are within the same band). This prices the result
/// payload leaving the engine, a different hop from the S3→engine GET traffic
/// (which is intra-region and byte-free on AWS) — so it never overlaps the
/// request/compute legs. Operators serving same-region clients override it to
/// 0 via `INFINO_BENCH_COST_EGRESS_USD_PER_GB`.
const DEFAULT_EGRESS_USD_PER_GB: f64 = 0.09;

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
/// Default warm fraction of the blended monthly read line (the rest pay
/// the cold per-query cost). Env-overridable via
/// `INFINO_BENCH_COST_WARM_FRACTION` (0 < f ≤ 1) so pricing scenarios can
/// flex the assumed hit rate without recompiling — the frequency knob;
/// the warm/cold rate gap the summary prints is the magnitude it
/// multiplies against.
const DEFAULT_SUMMARY_READ_WARM_FRACTION: f64 = 0.95;
/// Padding on the per-query RAM-hold window: a query holds the resident set
/// a little longer than its own p50 (dispatch, response write, scheduler
/// slack between overlapped queries), so the hold is billed at fudge × p50.
/// Residency is otherwise billed strictly per query served — never as a
/// standing calendar-hours line. Bench cost model only: real customer
/// metering must record the exact measured hold time and put any padding
/// in the PRICE, never in the reported quantity.
const QUERY_RAM_HOLD_FUDGE: f64 = 2.0;
/// Average hours in a calendar month (365.25 days × 24 h ÷ 12). Used only by
/// the provisioned-occupancy block to price a whole reserved node per month —
/// the marginal model never bills calendar hours (residency stays inside each
/// query's RAM-hold leg).
const HOURS_PER_MONTH: f64 = 730.5;
/// Default replication factor for the provisioned-occupancy framing: an
/// R-way HA placement where each replica keeps its own independent local
/// cache, so NVMe and RAM occupancy multiply by R rather than being shared.
/// The bench itself measures ONE node — the per-replica column is the
/// measured basis and the ×R column applies this multiplier visibly.
/// Override via `INFINO_BENCH_COST_REPLICAS`.
const DEFAULT_OCCUPANCY_REPLICAS: u32 = 2;

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
            // Storage-optimized reference shape: the keep-warm tier of an
            // object-storage-native service is provisioned for NVMe, so
            // occupancy shares price against a storage-dense node. The
            // previous default (c7gd.2xlarge, 237 GB NVMe) made cache
            // occupancy look ~20x more expensive per GiB than the nodes a
            // real deployment would buy. 8 vCPU / 64 GiB / 2x2500 GB NVMe,
            // us-east-1 Linux on-demand (verified 2026-07-31).
            name: "i3en.2xlarge".into(),
            vcpu: 8,
            ram_gib: 64.0,
            nvme_gb: 5000.0,
            usd_per_hour: 0.904,
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

    /// Whole-node dollars for one calendar month — the reserved-capacity
    /// rate the provisioned-occupancy block prices shares against.
    fn usd_per_month(&self) -> f64 {
        self.usd_per_hour * HOURS_PER_MONTH
    }

    /// Fraction of the instance's local NVMe a cache footprint occupies.
    /// NVMe capacity is quoted in decimal GB (matching cloud SKU listings).
    fn nvme_share(&self, bytes: u64) -> f64 {
        bytes as f64 / BYTES_PER_GB / self.nvme_gb
    }

    /// Fraction of the node's CPU an assumed monthly query load keeps busy:
    /// queries/month × billed vCPU·s per query ÷ the node's total vCPU·s in
    /// a month. "Billed" vCPU·s is `per_query_vcpu_seconds` — the binding of
    /// measured CPU and the RAM-hold leg — so the occupancy view and the
    /// marginal model reconcile on the same per-query quantity.
    fn cpu_share_at_load(&self, billed_vcpu_s_per_query: f64, queries_per_month: f64) -> f64 {
        queries_per_month * billed_vcpu_s_per_query
            / (f64::from(self.vcpu.max(1)) * HOURS_PER_MONTH * SECS_PER_HOUR)
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
    /// Median object-store GETs of the first cold search (process-default
    /// meter delta around the search call). Prices the cold request leg.
    pub search_get_count: u64,
    /// Median downloaded bytes of the first cold search.
    pub search_get_bytes: u64,
}

/// Warm query timing, measured on-CPU seconds, and the return-payload size
/// (rows + logical value bytes) of the query's realistic result — the
/// quantity network egress is priced from. Payload is cache-state-independent
/// (the same bytes leave the engine warm or cold), so this warm figure is also
/// the egress payload for the cold path.
#[derive(Clone)]
pub struct WarmQueryCost {
    pub name: String,
    pub p50_s: f64,
    pub cpu_s: Option<f64>,
    pub payload_rows: u64,
    pub payload_bytes: u64,
}

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
    /// Recall@10 for this state's default config, when the modality measures
    /// one (vector). Drives the pivoted per-state serving table's quality
    /// column.
    pub recall: Option<f32>,
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
    /// Wall/CPU of [`Self::cold_second_query`] when the shared cold-store
    /// helper timed the steady window (median across samples).
    pub cold_second_wall_s: Option<f64>,
    pub cold_second_cpu_s: Option<f64>,
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
    /// Settled local NVMe disk-cache footprint (`DiskCacheStore` current
    /// bytes) sampled on the shared consumer after the steady-state serving
    /// battery. Includes the hidden vector-index table — it shares the same
    /// cache root and budget. Feeds the provisioned-occupancy NVMe rows;
    /// `None` (e.g. an in-memory cell with no disk cache) omits them,
    /// never guesses.
    pub disk_cache_bytes: Option<u64>,
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
    /// Whether this cell runs the full maintenance lifecycle (pre-drain →
    /// drain → post-drain → delta → post-delta → compact → post-compact).
    /// Ledger rows for drain / delta / optimize / pre-drain cold always
    /// render on such a cell — as "NOT METERED" when the harness failed
    /// to measure them — and never render elsewhere. Named for the
    /// vector cell that introduced the shape; FTS/SQL set this too when
    /// they run the same phase sequence (drain is a no-op without a
    /// hidden vector index).
    pub vector_cell: bool,
    /// Assumed retention for the capacity line (GB-months). Default 1 month.
    pub storage_months: Option<f64>,
    /// Whether a cold `open` is a one-time table/namespace open that is
    /// amortized across every query (supertable: manifest load + consumer
    /// setup, paid once), rather than per-query latency. For a single
    /// superfile the open is part of each cold read, so this is `false`.
    pub cold_open_amortized: bool,
    /// Per-group serving breakdown: `(group_label, query_names)`. When set,
    /// the Serving table and the monthly read line are priced from the
    /// full query battery (`warm` / `cold`, the same measurement the
    /// per-shape search table reports), aggregated per group as the
    /// arithmetic mean of the group's per-query p50s and per-query cost —
    /// so the two tables reconcile by construction. `None` keeps the
    /// per-lifecycle-state serving rows (the vector cell's single-config
    /// probe). Text cells (FTS/SQL) set this; vector leaves it `None`.
    pub serving_groups: Option<&'a [(&'a str, &'a [&'a str])]>,
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

/// A capacity share as a percentage, with adaptive precision below 0.1% so
/// a tiny CPU share (e.g. 0.06%) never collapses to a meaningless "0.0%" —
/// same rationale as [`usd`].
fn fmt_share(share: f64) -> String {
    let pct = share * 100.0;
    if pct == 0.0 {
        return "0%".into();
    }
    if pct >= 0.1 {
        return format!("{pct:.1}%");
    }
    let decimals = ((-pct.log10()).ceil() as usize + 1).min(9);
    format!("{pct:.decimals$}%")
}

/// The binding provisioned resource: the largest MEASURED share with its
/// label. Unmeasured shares are skipped, never treated as 0 (a missing
/// measurement must not "lose" the max and misname the binding resource);
/// `None` only when nothing was measured.
fn binding_share(shares: &[(&'static str, Option<f64>)]) -> Option<(&'static str, f64)> {
    shares
        .iter()
        .filter_map(|(label, share)| share.map(|s| (*label, s)))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
}

/// Per-query cost with both scales visible — prevents comparing $/open to $/1M.
fn usd_per_query_both_scales(per_query: f64) -> String {
    format!("{}/query ({})", usd(per_query), usd_per_million(per_query))
}

/// Speed per dollar: `1 ÷ (p50 seconds × $/query)`. A joint figure of
/// merit — getting faster OR cheaper raises it (a plain latency÷cost
/// ratio rewards slowness), so higher is always better and the cell is
/// delta-tracked.
fn speed_per_usd(per_query_usd: f64, latency_s: f64) -> f64 {
    1.0 / (latency_s.max(f64::MIN_POSITIVE) * per_query_usd.max(f64::MIN_POSITIVE))
}

/// `1/(s·$)` cell rendered at count scale (`11.7K`), delta-tracked
/// higher-is-better.
fn speed_per_usd_cell(per_query_usd: f64, latency_s: f64) -> Cell {
    let v = speed_per_usd(per_query_usd, latency_s);
    metric(v, fmt_count(v as usize), Better::Higher)
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

/// Replication factor for the provisioned-occupancy block. Zero or garbage
/// falls back to the default — R=0 would print a $0 keep-warm bill, which is
/// never a measurement.
fn parse_replicas(raw: Option<&str>) -> u32 {
    raw.and_then(|s| s.parse::<u32>().ok())
        .filter(|&r| r > 0)
        .unwrap_or(DEFAULT_OCCUPANCY_REPLICAS)
}

/// R for the occupancy block, env-overridable like [`storage_months`] (a
/// deployment-topology knob, not an instance attribute, so it is read here
/// and not in `Instance::from_env`).
fn occupancy_replicas() -> u32 {
    static REPLICAS: OnceLock<u32> = OnceLock::new();
    *REPLICAS
        .get_or_init(|| parse_replicas(std::env::var("INFINO_BENCH_COST_REPLICAS").ok().as_deref()))
}

/// Parse an `INFINO_BENCH_COST_WARM_FRACTION` override. Accepts only
/// 0 < f ≤ 1; anything else (unset, garbage, 0, negative, >1, NaN) falls
/// back to the default — a nonsense hit rate must never silently shape
/// the blend.
fn parse_warm_fraction(raw: Option<&str>) -> f64 {
    raw.and_then(|s| s.parse::<f64>().ok())
        .filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0)
        .unwrap_or(DEFAULT_SUMMARY_READ_WARM_FRACTION)
}

/// Warm fraction for the blended read line, env-overridable like
/// [`occupancy_replicas`] (a workload assumption, not an instance
/// attribute).
fn summary_warm_fraction() -> f64 {
    static FRACTION: OnceLock<f64> = OnceLock::new();
    *FRACTION.get_or_init(|| {
        parse_warm_fraction(
            std::env::var("INFINO_BENCH_COST_WARM_FRACTION")
                .ok()
                .as_deref(),
        )
    })
}

/// The tenant's composed serving cost per month: marginal work (storage +
/// blended reads + writes + maintenance, egress excluded as pass-through)
/// plus the keep-warm floor (idle-retained NVMe × R) that buys the blend's
/// warm-hit rate. `None` floor (NVMe cache unmeasured) yields `None` — the
/// composed number is withheld rather than silently missing its floor.
fn serving_cogs_month(marginal_ex_egress: f64, idle_floor_total: Option<f64>) -> Option<f64> {
    idle_floor_total.map(|floor| marginal_ex_egress + floor)
}

/// Network egress rate ($/GB out), env-overridable like [`storage_months`]
/// (it is a rate-card rate, not an instance attribute, so it is read here and
/// not in `Instance::from_env`). Set `INFINO_BENCH_COST_EGRESS_USD_PER_GB=0`
/// for a same-region client where transfer is free.
fn egress_usd_per_gb() -> f64 {
    static RATE: OnceLock<f64> = OnceLock::new();
    *RATE.get_or_init(|| {
        std::env::var("INFINO_BENCH_COST_EGRESS_USD_PER_GB")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(DEFAULT_EGRESS_USD_PER_GB)
    })
}

/// Egress dollars for one query's return payload (decimal-GB priced, matching
/// storage). This is the ONLY leg that prices the returned result bytes; the
/// object-store `get_bytes` counted elsewhere is internal S3→engine fetch, a
/// separate hop, so the two never double-count.
fn egress_usd(payload_bytes: f64) -> f64 {
    payload_bytes / BYTES_PER_GB * egress_usd_per_gb()
}

/// Marks a serving family that must stay OUT of the bounded-result
/// monthly mean and be billed per event instead. Two classes qualify,
/// matched on the family label the runners construct:
///
/// * **Bulk row sets** — result size scales with the match set
///   (`O(selectivity × corpus)`), so the cost is per GB returned and a
///   per-query rate is meaningless.
/// * **Scan-backed aggregates** — whole-corpus CPU per query. Averaged
///   into the blend at equal weight they dominate it: at 10M a scan
///   aggregate prices in the $100s/1M while the selective shapes sit at
///   $1–5/1M, dragging the "SQL COGS" headline to ~$160/1M when the
///   selective serving mix it is quoted against costs a few dollars.
///   Scan work is real Infino capability — priced per event, according
///   to the work, never silently averaged into retrieval pricing.
fn is_bulk_group(label: &str) -> bool {
    label.starts_with("Bulk") || label.starts_with("Aggregates — scan-backed")
}

/// "$X/1M" cell text for one query's egress, from its payload bytes. Battery
/// tables call this so every query row carries its own egress cost.
pub fn egress_cell_per_million(payload_bytes: u64) -> String {
    format!("{}/1M", usd(egress_usd(payload_bytes as f64) * PER_MILLION))
}

/// "$X/1M" cell text for one warm query's FULL cost — compute (binding
/// CPU/RAM leg) + egress on its payload. "—" when its CPU was unsampled
/// (never a $0 guess).
pub fn warm_cell_per_million(
    cpu_s: Option<f64>,
    p50_s: f64,
    resident_bytes: u64,
    payload_bytes: u64,
) -> String {
    match cpu_s {
        Some(cpu) => {
            let per_q = Instance::current().per_query_usd(cpu, p50_s, resident_bytes)
                + egress_usd(payload_bytes as f64);
            format!("{}/1M", usd(per_q * PER_MILLION))
        }
        None => "—".into(),
    }
}

/// "$X/1M" cell text for one cold query's FULL cost — compute over the
/// same-config warm window (the RAM-hold convention the serving table uses) +
/// GET requests + egress on its payload. "—" when its CPU was unsampled.
pub fn cold_cell_per_million(
    search_cpu_s: Option<f64>,
    warm_window_s: f64,
    resident_bytes: u64,
    get_count: u64,
    payload_bytes: u64,
) -> String {
    match search_cpu_s {
        Some(cpu) => {
            let per_q = Instance::current().per_query_usd(cpu, warm_window_s, resident_bytes)
                + get_count as f64 * USD_PER_GET
                + egress_usd(payload_bytes as f64);
            format!("{}/1M", usd(per_q * PER_MILLION))
        }
        None => "—".into(),
    }
}

/// Full DIRECT cost of one write op, in dollars: compute billed at the
/// binding leg — pool CPU or peak-RSS share of RAM, whichever is larger —
/// over the op's wall window, plus its object-store request dollars.
/// `None` when the op's CPU was unsampled (never a $0 guess — the same
/// rule the phase and query cells follow).
///
/// "Direct" is deliberate: deferred maintenance the write will later
/// cause (hidden-index drain merges, compaction) is excluded here and
/// is already inside the amortized `write_per_million_docs` figure.
/// Per-op cells exist to compare write shapes against each other on the
/// work the op itself performed.
pub fn write_op_usd(
    cpu_s: Option<f64>,
    wall_s: f64,
    peak_rss_bytes: Option<u64>,
    io: &ObjectStoreMeter,
) -> Option<f64> {
    cpu_s.map(|cpu| {
        let inst = Instance::current();
        inst.compute_usd(inst.phase_vcpu_seconds(cpu, wall_s, peak_rss_bytes)) + request_usd(io)
    })
}

/// "$X/write ($Y/1M)" cell text for one write op's direct cost.
pub fn write_cell(per_op: f64) -> String {
    format!("{}/write ({})", usd(per_op), usd_per_million(per_op))
}

/// Plain "$X" cell text for an already-scaled dollar figure (e.g. a
/// per-1M-rows value a caller computed from a per-op cost).
pub fn usd_text(v: f64) -> String {
    usd(v)
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
    // The write-compute figure is COMPLETE only if every write phase that ran
    // had its CPU sampled. An unsampled phase must read NOT METERED — the model
    // never back-fills it with a wall-clock guess (see the comment above) — so
    // a tier that skips CPU sampling (e.g. the in-memory superfile micro-bench,
    // which passes `ingest_cpu_s: None`) must not render `unwrap_or(0.0)` as a
    // misleading $0 build cost. Ingest always runs; drain/delta/optimize only
    // when their wall time is present.
    let write_compute_metered = c.ingest_cpu_s.is_some()
        && (c.store.drain_wall_s.is_none() || c.store.drain_cpu_s.is_some())
        && (c.store.delta_commit_wall_s.is_none() || c.store.delta_commit_cpu_s.is_some())
        && (c.store.compaction_wall_s.is_none() || c.store.compaction_cpu_s.is_some());
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
    // Two distinct quantities, kept apart: the rate card prices the whole
    // stated retention (and says so), while the monthly summary bills ONE
    // month. Multiplying retention into the monthly line would charge N
    // months of storage inside a $/month column and its total.
    let storage_retention_usd = gb_months * USD_PER_GB_MONTH;
    let storage_month = stored_gb * USD_PER_GB_MONTH;

    // ---- Warm query battery (priced from MEASURED on-CPU seconds) ----
    // Only entries with a sampled cpu are priced; an unmetered warm query is
    // omitted from the battery rather than back-filled with a wall guess.
    let warm_costs: Vec<(f64, f64, String)> = c
        .warm
        .iter()
        .filter_map(|w| {
            w.cpu_s.map(|cpu| {
                let per_q = inst.per_query_usd(cpu, w.p50_s, c.resident_anon_bytes);
                (per_q, w.p50_s, w.name.clone())
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
            .find(|w| w.name == name)
            .or_else(|| c.warm.first())
            .map(|w| w.p50_s)
    };

    // Per-query cold dollars use the *steady* cold window only (cold_second),
    // never the first-query metadata warmup. Object-store `get_bytes` are
    // internal fetch volume — not customer egress — and are not priced here.
    //
    // Three distinct outcomes, kept apart rather than collapsed into one
    // `Option<f64>`: a steady sample can be fully priced (compute + requests
    // from the SAME steady window), request-only (compute unsampled — must
    // say so, never silently $0), or entirely absent (falls back to the
    // first-cold latency, which must be labeled "first-cold", never "steady").
    let cold_second_io = c.store.cold_second_query;
    enum SteadyColdPrice {
        Full {
            per_q: f64,
            wall_s: f64,
            /// Billed compute of the steady cold query (max of measured CPU
            /// and the RAM-hold leg, no request dollars) — the quantity the
            /// occupancy block's CPU share is built from.
            vcpu_s: f64,
        },
        RequestsOnly {
            usd: f64,
        },
        None,
    }
    let steady_cold_price = match (
        cold_second_io,
        c.store.cold_second_cpu_s,
        c.store.cold_second_wall_s,
    ) {
        // CPU + wall must both be from the steady sample; never borrow a
        // warm / first-query window into the steady price.
        (Some(io), Some(cpu), Some(window)) => SteadyColdPrice::Full {
            per_q: inst.per_query_usd(cpu, window, c.resident_anon_bytes) + request_usd(&io),
            wall_s: window,
            vcpu_s: inst.per_query_vcpu_seconds(cpu, window, c.resident_anon_bytes),
        },
        (Some(io), _, _) => SteadyColdPrice::RequestsOnly {
            usd: request_usd(&io),
        },
        (None, _, _) => SteadyColdPrice::None,
    };

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
                usd(per_million_docs(storage_retention_usd)),
                usd_per_gb(USD_PER_GB_MONTH),
            )),
        ],
        vec![
            text(write_label),
            text(if write_compute_metered {
                format!(
                    "{} compute + {} requests → {} total ({}/1M docs)",
                    usd(write_compute),
                    usd(write_requests),
                    usd(write_total),
                    usd(write_per_million_docs),
                )
            } else {
                // A write phase ran but its CPU wasn't sampled: show requests
                // only and flag compute NOT METERED, never a $0 that reads as
                // "the build was free".
                format!(
                    "compute NOT METERED + {} requests ({}/1M docs requests)",
                    usd(write_requests),
                    usd(per_million_docs(write_requests)),
                )
            }),
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
        match &steady_cold_price {
            SteadyColdPrice::Full { per_q, wall_s, .. } => {
                let io = cold_second_io.expect("Full variant implies cold_second_io");
                rate_rows.push(vec![
                    text("Cold query (CPU + requests, steady)"),
                    text(format!(
                        "{} queries — {} GET/query, {}/query fetched ({} search, {})",
                        usd_per_million(*per_q),
                        io.get_count,
                        fmt_bytes(io.get_bytes),
                        fmt_time(*wall_s * 1e9),
                        q.name,
                    )),
                ]);
            }
            SteadyColdPrice::RequestsOnly { usd: req_usd } => {
                let io = cold_second_io.expect("RequestsOnly variant implies cold_second_io");
                rate_rows.push(vec![
                    text("Cold query (requests only, steady — compute NOT METERED)"),
                    text(format!(
                        "{} queries — {} GET/query, {}/query fetched ({}; steady compute unsampled)",
                        usd_per_million(*req_usd),
                        io.get_count,
                        fmt_bytes(io.get_bytes),
                        q.name,
                    )),
                ]);
            }
            SteadyColdPrice::None => {
                rate_rows.push(vec![
                    text("Cold query (latency only — requests not metered)"),
                    text(format!(
                        "{} open + {} search ({}, first-cold; steady not sampled)",
                        fmt_time(q.open_s * 1e9),
                        fmt_time(q.search_s * 1e9),
                        q.name,
                    )),
                ]);
            }
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
            (Some(q), Some(r)) => Some(merge_background_fill(&q, &r)),
            (Some(q), None) => Some(background_fill_meter(&q)),
            (None, Some(r)) => Some(background_fill_meter(&r)),
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
            // The array is fixed at 4 slots; a lifecycle with fewer states
            // (FTS/SQL run pre-compact + post-compact only) leaves the tail
            // slots unlabeled — skip them rather than render empty rows.
            let Some(label) = state.io.label else {
                continue;
            };
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
                (Some(q), Some(r)) => Some(merge_background_fill(&q, &r)),
                (Some(q), None) => Some(background_fill_meter(&q)),
                (None, Some(r)) => Some(background_fill_meter(&r)),
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
    // Filtered search (~10% allow-set) is a vector-only battery; FTS/SQL
    // never run it, so only cells that carry filtered data render the row
    // (prevents a spurious NOT-METERED row in the text lifecycles).
    if c.store.filtered_query.is_some() || c.store.filtered_query_iters > 0 {
        averaged_row(
            &mut io_rows,
            "Filtered warm (~10%)",
            c.store.filtered_query,
            c.store.filtered_query_iters,
        );
    }
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
    // A phase's row shows COMPUTE only in its own cell, then the matching
    // REQUEST leg (from the SAME window's I/O ledger row) and the sum, so
    // the row reconciles with the Serving/Monthly total inline — a reader
    // never has to cross-reference a different table to see the full price
    // (the confusion a compute-only "Cost" column invited).
    let phase_row = |label: String,
                     wall_s: f64,
                     peak_rss: Option<u64>,
                     cpu_s: Option<f64>,
                     req_usd: f64|
     -> Vec<Cell> {
        let Some(cpu) = cpu_s else {
            return vec![
                text(label),
                text(fmt_wall_seconds(wall_s)),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text(usd(req_usd)),
                text("N/A (compute unmetered)"),
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
            text(usd(usd_v)),
            text(usd(req_usd)),
            metric(usd_v + req_usd, usd(usd_v + req_usd), Better::Lower),
        ]
    };
    let mut compute_rows = vec![phase_row(
        "Ingest".into(),
        c.ingest_wall_s,
        c.ingest_peak_rss_bytes,
        c.ingest_cpu_s,
        ingest_req_usd,
    )];
    if c.store.drain_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Drain".to_string(),
            drain_wall_s,
            c.store.drain_peak_rss_bytes,
            c.store.drain_cpu_s,
            drain_req_usd,
        ));
    } else if c.vector_cell {
        compute_rows.push(phase_row("Drain".to_string(), 0.0, None, None, 0.0));
    }
    if c.store.delta_commit_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Delta commit".to_string(),
            delta_wall_s,
            c.store.delta_commit_peak_rss_bytes,
            c.store.delta_commit_cpu_s,
            delta_req_usd,
        ));
    }
    if c.store.compaction_wall_s.is_some() {
        compute_rows.push(phase_row(
            "Optimize".to_string(),
            compaction_wall_s,
            c.store.compaction_peak_rss_bytes,
            c.store.compaction_cpu_s,
            compaction_req_usd,
        ));
    } else if c.vector_cell {
        compute_rows.push(phase_row("Optimize".to_string(), 0.0, None, None, 0.0));
    }
    if query_states.is_empty()
        && let Some(q) = anchor_cold
    {
        let open_label = format!("Open — {}", q.name);
        // Table open is compute-bound (manifest parse + reader CRC: measured
        // cpu ≈ wall), so it's priced from its MEASURED on-CPU seconds. NOT
        // METERED (never latency × share) when unsampled.
        let open_req = c.store.cold_open.map(|io| request_usd(&io)).unwrap_or(0.0);
        compute_rows.push(match q.open_cpu_s {
            Some(cpu) => {
                // Same floor every other phase uses: on-CPU ticks are
                // 10ms-quantized (CLK_TCK), so a genuinely fast open can
                // measure exactly 0.0 ticks and must not price as free
                // compute — bind to the RAM-hold leg like every other row.
                let billed = cpu.max(inst.ram_leg(q.open_s, Some(c.resident_anon_bytes)));
                let open_usd = inst.compute_usd(billed);
                vec![
                    text(open_label),
                    text(fmt_wall_seconds(q.open_s)),
                    text(fmt_vcpu_seconds(cpu)),
                    text(fmt_bytes(c.resident_anon_bytes)),
                    text("CPU"),
                    text(usd(open_usd)),
                    text(usd(open_req)),
                    metric(open_usd + open_req, usd(open_usd + open_req), Better::Lower),
                ]
            }
            None => vec![
                text(open_label),
                text(fmt_wall_seconds(q.open_s)),
                text("N/A"),
                text("N/A"),
                text("N/A"),
                text("N/A (compute unmetered)"),
                text(usd(open_req)),
                text("N/A (compute unmetered)"),
            ],
        });
        // Cold search CPU: MEASURED on-CPU during the search window (decompress,
        // decode, scoring — not copied from warm), with the RAM leg over the
        // warm-scale compute window. NOT METERED when unsampled.
        let cold_req = q.search_get_count as f64 * USD_PER_GET;
        compute_rows.push(match q.search_cpu_s {
            Some(cpu) => {
                let window = warm_window_for(&q.name).unwrap_or(0.0);
                let ram = inst.query_ram_leg(window, c.resident_anon_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu, window, c.resident_anon_bytes);
                let per_q = inst.compute_usd(vcpu);
                let total_q = per_q + cold_req;
                let binding = if ram > cpu { "RAM" } else { "CPU" };
                vec![
                    text(format!("Cold — {}", q.name)),
                    text(fmt_time(q.search_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu)),
                    text(fmt_bytes(c.resident_anon_bytes)),
                    text(binding),
                    text(usd_per_query_both_scales(per_q)),
                    text(usd_per_query_both_scales(cold_req)),
                    metric(
                        total_q * PER_MILLION,
                        usd_per_query_both_scales(total_q),
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
                text("N/A (compute unmetered)"),
                text(usd_per_query_both_scales(cold_req)),
                text("N/A (compute unmetered)"),
            ],
        });
    }
    if query_states.is_empty()
        && let Some(WarmQueryCost {
            name, p50_s, cpu_s, ..
        }) = c
            .warm
            .iter()
            .find(|w| w.name == "ten_term_or")
            .or_else(|| c.warm.first())
    {
        // Warm query priced from MEASURED on-CPU seconds (per-vCPU) with the
        // RAM leg over its own p50 window. NOT METERED when unsampled.
        // No I/O meter exists for this single-representative warm row (an
        // older, mostly-superseded path); a genuinely warm query is 0-GET by
        // design (`wait_until_warm` settles fills before this window), so
        // $0 here is a true zero, not an omission.
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
                    text(usd_per_query_both_scales(per_q)),
                    text("$0 (warm assumed 0-GET)"),
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
                text("N/A (compute unmetered)"),
                text("$0 (warm assumed 0-GET)"),
                text("N/A (compute unmetered)"),
            ],
        });
    }
    for state in &query_states {
        let label = state.io.label.expect("filtered query state has a label");
        let open_req = state.io.cold_open.as_ref().map(request_usd).unwrap_or(0.0);
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
                    text(usd(usd_v)),
                    text(usd(open_req)),
                    metric(usd_v + open_req, usd(usd_v + open_req), Better::Lower),
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
                text("N/A (compute unmetered)"),
                text(usd(open_req)),
                text("N/A (compute unmetered)"),
            ],
        });
        let cold_query_req = state.io.cold_query.as_ref().map(request_usd).unwrap_or(0.0);
        compute_rows.push(match (state.cold_query_s, state.cold_query_cpu_s) {
            (Some(wall_s), Some(cpu_s)) => {
                let warm_window = state.warm_p50_s.unwrap_or(0.0);
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let ram = inst.query_ram_leg(warm_window, ram_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes);
                let per_q = inst.compute_usd(vcpu);
                let total_q = per_q + cold_query_req;
                vec![
                    text(format!("Cold 1st (warmup) — {label}")),
                    text(fmt_time(wall_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu_s)),
                    text(state.serving_ram_label(c.resident_anon_bytes)),
                    text(if ram > cpu_s { "RAM" } else { "CPU" }),
                    text(usd_per_query_both_scales(per_q)),
                    text(usd_per_query_both_scales(cold_query_req)),
                    metric(
                        total_q * PER_MILLION,
                        usd_per_query_both_scales(total_q),
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
                text("N/A (compute unmetered)"),
                text(usd_per_query_both_scales(cold_query_req)),
                text("N/A (compute unmetered)"),
            ],
        });
        if let (Some(wall_s), Some(cpu_s)) = (state.cold_second_s, state.cold_second_cpu_s) {
            let warm_window = state.warm_p50_s.unwrap_or(0.0);
            let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
            let ram = inst.query_ram_leg(warm_window, ram_bytes);
            let vcpu = inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes);
            let per_q = inst.compute_usd(vcpu);
            let cold_second_req = state
                .io
                .cold_second
                .as_ref()
                .map(request_usd)
                .unwrap_or(0.0);
            let total_q = per_q + cold_second_req;
            compute_rows.push(vec![
                text(format!("Cold 2nd (steady) — {label}")),
                text(fmt_time(wall_s * 1e9)),
                text(fmt_vcpu_seconds(cpu_s)),
                text(state.serving_ram_label(c.resident_anon_bytes)),
                text(if ram > cpu_s { "RAM" } else { "CPU" }),
                text(usd_per_query_both_scales(per_q)),
                text(usd_per_query_both_scales(cold_second_req)),
                metric(
                    total_q * PER_MILLION,
                    usd_per_query_both_scales(total_q),
                    Better::Lower,
                ),
            ]);
        }
        let warm_req = state
            .io
            .warm
            .as_ref()
            .map(|io| request_usd(io) / state.io.warm_iters.max(1) as f64)
            .unwrap_or(0.0);
        compute_rows.push(match (state.warm_p50_s, state.warm_cpu_s) {
            (Some(p50_s), Some(cpu_s)) => {
                let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
                let ram = inst.query_ram_leg(p50_s, ram_bytes);
                let vcpu = inst.per_query_vcpu_seconds(cpu_s, p50_s, ram_bytes);
                let per_q = inst.compute_usd(vcpu);
                let total_q = per_q + warm_req;
                vec![
                    text(format!("Warm — {label}")),
                    text(fmt_time(p50_s * 1e9)),
                    text(fmt_vcpu_seconds(cpu_s)),
                    text(state.serving_ram_label(c.resident_anon_bytes)),
                    text(if ram > cpu_s { "RAM" } else { "CPU" }),
                    text(usd_per_query_both_scales(per_q)),
                    text(usd_per_query_both_scales(warm_req)),
                    metric(
                        total_q * PER_MILLION,
                        usd_per_query_both_scales(total_q),
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
                text("N/A (compute unmetered)"),
                text(usd_per_query_both_scales(warm_req)),
                text("N/A (compute unmetered)"),
            ],
        });
    }
    let compute_ledger = Block {
        subtitle:
            "Compute — actual CPU time and resident RAM; binding determines cost. Total adds \
             the request leg from the same window's I/O ledger row, so each row reconciles \
             with Serving/Monthly without cross-referencing another table."
                .into(),
        headers: vec![
            "Phase".into(),
            "Wall / p50".into(),
            "CPU (s)".into(),
            "RAM".into(),
            "Binding".into(),
            "Compute".into(),
            "Requests".into(),
            "Total".into(),
        ],
        rows: compute_rows,
    };

    // ---- Block 4: serving ----
    let mut serving_rows: Vec<Vec<Cell>> = Vec::new();
    // Steady-state per-query dollars for the monthly summary: the LAST
    // populated query state (post-compact when the lifecycle ran) is the
    // shape a long-lived table serves.
    // Monthly per-query rates are built from BOUNDED-result shapes only;
    // unbounded (bulk) shapes are collected separately and billed per event.
    let mut monthly_bounded_payloads: Vec<u64> = Vec::new();
    let mut monthly_bulk_shapes: Vec<(String, f64, u64)> = Vec::new();
    let mut steady_warm: Option<(String, f64)> = None;
    let mut steady_cold: Option<(String, f64)> = None;
    // Billed compute (vCPU·s) of the same steady warm/cold queries the dollar
    // figures above are built from — set at the same branches, blended the
    // same way — so the occupancy block's CPU share prices exactly the load
    // the monthly read line bills (the dollar figures can't stand in: the
    // cold side folds in GET request dollars, which are not compute).
    let mut steady_warm_vcpu_s: Option<f64> = None;
    let mut steady_cold_vcpu_s: Option<f64> = None;
    if let Some(groups) = c.serving_groups {
        // Per-group serving priced from the full query battery — the SAME
        // measurement the per-shape search table reports, so the two tables
        // reconcile by construction. Each row is the arithmetic mean of the
        // group's per-query p50s and per-query cost (compute + cold request
        // leg from the metered per-query GETs). The monthly blend below uses
        // the battery-wide arithmetic mean.
        //
        // RAM-leg residency is the ENGINE-ONLY set (pinned heap + settled
        // file cache, harness heap excluded) of the last populated routing
        // state — the steady-state layout a long-lived table serves — so the
        // serving $ matches the compute ledger's rows for the same shapes
        // rather than being inflated by whole-process anon RSS. Falls back to
        // whole-process anon only when no state sampled the anon/file split.
        let resident = query_states
            .last()
            .map(|s| s.serving_resident_bytes(c.resident_anon_bytes))
            .unwrap_or(c.resident_anon_bytes);
        // Each returns (p50_seconds, per_query_usd, billed_vcpu_s,
        // payload_bytes). Payload is the returned result size
        // (cache-independent), the egress quantity; billed vCPU·s is the
        // compute-only quantity behind the dollars (no request leg).
        let warm_per_q = |name: &str| -> Option<(f64, f64, f64, u64)> {
            c.warm.iter().find(|w| w.name == name).and_then(|w| {
                w.cpu_s.map(|cpu| {
                    (
                        w.p50_s,
                        inst.per_query_usd(cpu, w.p50_s, resident),
                        inst.per_query_vcpu_seconds(cpu, w.p50_s, resident),
                        w.payload_bytes,
                    )
                })
            })
        };
        let cold_per_q = |name: &str| -> Option<(f64, f64, f64, u64)> {
            let cq = c.cold?.iter().find(|q| q.name == name)?;
            let cpu = cq.search_cpu_s?;
            // Cold holds the resident set for ~its warm window (bytes local);
            // the rest of the cold p50 is off-CPU I/O wait. Cold returns the
            // same result bytes as warm, so payload — and the RAM-hold
            // window — come from the matching warm shape; a cold-only run
            // with no warm counterpart has no measured payload for this
            // shape and is skipped, rather than priced from a fabricated
            // 0-byte payload and the I/O-wait-inclusive cold wall.
            let warm = c.warm.iter().find(|w| w.name == name)?;
            let per_q = inst.per_query_usd(cpu, warm.p50_s, resident)
                + cq.search_get_count as f64 * USD_PER_GET;
            Some((
                cq.search_s,
                per_q,
                inst.per_query_vcpu_seconds(cpu, warm.p50_s, resident),
                warm.payload_bytes,
            ))
        };
        let mut all_warm_usd: Vec<f64> = Vec::new();
        let mut all_cold_usd: Vec<f64> = Vec::new();
        let mut all_warm_vcpu_s: Vec<f64> = Vec::new();
        let mut all_cold_vcpu_s: Vec<f64> = Vec::new();
        // Family row: p50 / payload / egress are the family means; the final
        // per-query dollars are the FULL serve cost — compute + requests +
        // egress on the returned bytes — so the row is the complete unit cost
        // of that query class.
        let group_row = |rows: &mut Vec<Vec<Cell>>,
                         label: &str,
                         kind: &str,
                         samples: &[(f64, f64, f64, u64)]| {
            if samples.is_empty() {
                return;
            }
            let n = samples.len() as f64;
            let mean_p50 = samples.iter().map(|(p, _, _, _)| p).sum::<f64>() / n;
            let mean_usd = samples.iter().map(|(_, u, _, _)| u).sum::<f64>() / n;
            let mean_payload = samples.iter().map(|(_, _, _, b)| *b).sum::<u64>() as f64 / n;
            let egress = egress_usd(mean_payload);
            let total = mean_usd + egress;
            let qpu = 1.0 / total.max(f64::MIN_POSITIVE);
            rows.push(vec![
                text(format!(
                    "{label} — {kind} (mean of {} shapes)",
                    samples.len()
                )),
                text(fmt_time(mean_p50 * 1e9)),
                text(fmt_bytes(mean_payload as u64)),
                text(usd(egress * PER_MILLION)),
                metric(qpu, format!("{qpu:.0}"), Better::Higher),
                speed_per_usd_cell(total, mean_p50),
                metric(total * PER_MILLION, usd(total * PER_MILLION), Better::Lower),
            ]);
        };
        for (label, names) in groups {
            let warms: Vec<(f64, f64, f64, u64)> =
                names.iter().filter_map(|n| warm_per_q(n)).collect();
            let colds: Vec<(f64, f64, f64, u64)> =
                names.iter().filter_map(|n| cold_per_q(n)).collect();
            // BOUNDED shapes only feed the per-query monthly rates. A bulk
            // shape's result scales with the match set, so averaging it into a
            // per-query rate asserts that every query returns a full-corpus
            // dump — the mean of a 193 MiB scan and twenty ~KB lookups is a
            // number no query has. Bulk shapes are billed per event instead
            // (see the bulk rate lines in the monthly summary).
            if is_bulk_group(label) {
                monthly_bulk_shapes.extend(names.iter().filter_map(|n| {
                    let (_, warm_usd, _, payload) = warm_per_q(n)?;
                    Some((n.to_string(), warm_usd, payload))
                }));
            } else {
                all_warm_usd.extend(warms.iter().map(|(_, u, _, _)| *u));
                all_cold_usd.extend(colds.iter().map(|(_, u, _, _)| *u));
                all_warm_vcpu_s.extend(warms.iter().map(|(_, _, v, _)| *v));
                all_cold_vcpu_s.extend(colds.iter().map(|(_, _, v, _)| *v));
                monthly_bounded_payloads.extend(warms.iter().map(|(_, _, _, b)| *b));
            }
            group_row(&mut serving_rows, label, "warm", &warms);
            group_row(&mut serving_rows, label, "cold", &colds);
        }
        if !all_warm_usd.is_empty() {
            let mean = all_warm_usd.iter().sum::<f64>() / all_warm_usd.len() as f64;
            steady_warm = Some(("warm (bounded-result mean)".to_string(), mean));
        }
        if !all_cold_usd.is_empty() {
            let mean = all_cold_usd.iter().sum::<f64>() / all_cold_usd.len() as f64;
            steady_cold = Some(("cold (bounded-result mean)".to_string(), mean));
        }
        if !all_warm_vcpu_s.is_empty() {
            steady_warm_vcpu_s =
                Some(all_warm_vcpu_s.iter().sum::<f64>() / all_warm_vcpu_s.len() as f64);
        }
        if !all_cold_vcpu_s.is_empty() {
            steady_cold_vcpu_s =
                Some(all_cold_vcpu_s.iter().sum::<f64>() / all_cold_vcpu_s.len() as f64);
        }
    } else if query_states.is_empty() {
        serving_rows.extend(c.warm.iter().filter_map(|w| {
            let cpu = w.cpu_s?;
            let egress = egress_usd(w.payload_bytes as f64);
            let per_q = inst.per_query_usd(cpu, w.p50_s, c.resident_anon_bytes) + egress;
            let queries_per_usd = 1.0 / per_q.max(f64::MIN_POSITIVE);
            Some(vec![
                text(format!("{} — warm", w.name)),
                text(fmt_time(w.p50_s * 1e9)),
                text(fmt_bytes(w.payload_bytes)),
                text(usd(egress * PER_MILLION)),
                metric(
                    queries_per_usd,
                    format!("{queries_per_usd:.0}"),
                    Better::Higher,
                ),
                speed_per_usd_cell(per_q, w.p50_s),
                metric(per_q * PER_MILLION, usd(per_q * PER_MILLION), Better::Lower),
            ])
        }));
        if let Some(WarmQueryCost {
            name,
            p50_s,
            cpu_s: Some(cpu),
            ..
        }) = c
            .warm
            .iter()
            .find(|w| w.name == "ten_term_or")
            .or_else(|| c.warm.first())
        {
            let per_q = inst.per_query_usd(*cpu, *p50_s, c.resident_anon_bytes);
            steady_warm = Some((format!("warm ({name})"), per_q));
            steady_warm_vcpu_s =
                Some(inst.per_query_vcpu_seconds(*cpu, *p50_s, c.resident_anon_bytes));
        }
        if let Some(q) = anchor_cold {
            let payload = c
                .warm
                .iter()
                .find(|w| w.name == q.name)
                .map(|w| w.payload_bytes)
                .unwrap_or(0);
            let egress = egress_usd(payload as f64);
            match &steady_cold_price {
                SteadyColdPrice::Full {
                    per_q,
                    wall_s,
                    vcpu_s,
                } => {
                    let total = per_q + egress;
                    let queries_per_usd = 1.0 / total.max(f64::MIN_POSITIVE);
                    serving_rows.push(vec![
                        text(format!("{} — cold", q.name)),
                        text(fmt_time(*wall_s * 1e9)),
                        text(fmt_bytes(payload)),
                        text(usd(egress * PER_MILLION)),
                        metric(
                            queries_per_usd,
                            format!("{queries_per_usd:.0}"),
                            Better::Higher,
                        ),
                        speed_per_usd_cell(total, *wall_s),
                        metric(total * PER_MILLION, usd(total * PER_MILLION), Better::Lower),
                    ]);
                    steady_cold = Some((format!("cold ({})", q.name), *per_q));
                    steady_cold_vcpu_s = Some(*vcpu_s);
                }
                SteadyColdPrice::RequestsOnly { usd: req_usd } => {
                    // No steady wall-clock sample exists here — the first-cold
                    // wall (`q.search_s`) includes one-time metadata warmup and
                    // must never stand in for it, so latency/speed render "—"
                    // rather than a mislabeled number.
                    let total = req_usd + egress;
                    let queries_per_usd = 1.0 / total.max(f64::MIN_POSITIVE);
                    serving_rows.push(vec![
                        text(format!(
                            "{} — cold (requests only, compute NOT METERED)",
                            q.name
                        )),
                        text("—"),
                        text(fmt_bytes(payload)),
                        text(usd(egress * PER_MILLION)),
                        metric(
                            queries_per_usd,
                            format!("{queries_per_usd:.0}"),
                            Better::Higher,
                        ),
                        text("—"),
                        metric(total * PER_MILLION, usd(total * PER_MILLION), Better::Lower),
                    ]);
                    steady_cold = Some((format!("cold ({}, requests only)", q.name), *req_usd));
                }
                SteadyColdPrice::None => {}
            }
        }
    } else {
        // Vector returns id + score — a state-independent payload; show the
        // battery-mean result size on each lifecycle-state row.
        let vec_payload_bytes: Option<u64> = {
            let p: Vec<u64> = c.warm.iter().map(|w| w.payload_bytes).collect();
            if p.is_empty() {
                None
            } else {
                Some(p.iter().sum::<u64>() / p.len() as u64)
            }
        };
        let payload_cell = || {
            text(
                vec_payload_bytes
                    .map(fmt_bytes)
                    .unwrap_or_else(|| "—".into()),
            )
        };
        // Egress on the id+score result; folded into each state's full
        // per-query dollars (compute + requests + egress).
        let vec_egress = vec_payload_bytes
            .map(|b| egress_usd(b as f64))
            .unwrap_or(0.0);
        let egress_cell = || text(usd(vec_egress * PER_MILLION));
        // One pivoted row per lifecycle state: recall, payload, egress, warm
        // latency + full $, and both cold legs (1st = one-time metadata
        // warmup; steady = the per-query price cold traffic actually pays).
        for state in &query_states {
            let label = state.io.label.expect("filtered query state has a label");
            let ram_bytes = state.serving_resident_bytes(c.resident_anon_bytes);
            let mut row = vec![
                text(label.to_string()),
                text(
                    state
                        .recall
                        .map(|r| format!("{r:.3}"))
                        .unwrap_or_else(|| "—".into()),
                ),
                payload_cell(),
                egress_cell(),
            ];
            match (state.warm_p50_s, state.warm_cpu_s) {
                (Some(p50_s), Some(cpu_s)) => {
                    // Warm queries usually hit a hot cache (0 GETs), but not
                    // always — an undersized cache makes a "warm" query fetch.
                    // Price whatever the warm window actually metered instead
                    // of assuming zero.
                    let warm_req = match (state.io.warm, state.io.warm_iters) {
                        (Some(io), iters) if iters > 0 => request_usd(&io) / iters as f64,
                        _ => 0.0,
                    };
                    let per_q = inst.per_query_usd(cpu_s, p50_s, ram_bytes) + warm_req;
                    let total = per_q + vec_egress;
                    row.push(metric(p50_s * 1e9, fmt_time(p50_s * 1e9), Better::Lower));
                    row.push(text(format!("{}/1M", usd(total * PER_MILLION))));
                    steady_warm = Some((format!("warm — {label}"), per_q));
                    steady_warm_vcpu_s = Some(inst.per_query_vcpu_seconds(cpu_s, p50_s, ram_bytes));
                }
                _ => row.extend([text("—"), text("—")]),
            }
            let warm_window = state.warm_p50_s.unwrap_or(0.0);
            match (
                state.cold_query_s,
                state.cold_query_cpu_s,
                state.io.cold_query,
            ) {
                (Some(wall_s), Some(cpu_s), Some(io)) => {
                    let per_q =
                        inst.per_query_usd(cpu_s, warm_window, ram_bytes) + request_usd(&io);
                    row.push(text(format!(
                        "{} ({} GET)",
                        fmt_time(wall_s * 1e9),
                        io.get_count
                    )));
                    // Fallback steady leg when no second-query window was
                    // metered.
                    if state.io.cold_second.is_none() {
                        steady_cold =
                            Some((format!("cold — {label} ({} GET)", io.get_count), per_q));
                        steady_cold_vcpu_s =
                            Some(inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes));
                    }
                }
                _ => row.push(text("—")),
            }
            match (
                state.cold_second_s,
                state.cold_second_cpu_s,
                state.io.cold_second,
            ) {
                (Some(wall_s), Some(cpu_s), Some(io)) => {
                    let per_q =
                        inst.per_query_usd(cpu_s, warm_window, ram_bytes) + request_usd(&io);
                    let total = per_q + vec_egress;
                    row.push(metric(
                        wall_s * 1e9,
                        format!("{} ({} GET)", fmt_time(wall_s * 1e9), io.get_count),
                        Better::Lower,
                    ));
                    row.push(metric(
                        total * PER_MILLION,
                        format!("{}/1M", usd(total * PER_MILLION)),
                        Better::Lower,
                    ));
                    steady_cold = Some((
                        format!("cold steady — {label} ({} GET)", io.get_count),
                        per_q,
                    ));
                    steady_cold_vcpu_s =
                        Some(inst.per_query_vcpu_seconds(cpu_s, warm_window, ram_bytes));
                }
                _ => row.extend([text("—"), text("—")]),
            }
            serving_rows.push(row);
        }
    }
    let serving = Block {
        subtitle: if c.serving_groups.is_some() {
            "Serving — query latency and cost per query-family (arithmetic mean of the \
             family's per-shape p50s and per-query cost, from the same battery the search \
             table reports); 1/(s·$) is speed per dollar (1 ÷ (p50 seconds × $/query)), higher \
             is better."
                .to_string()
        } else {
            "Serving — query latency and cost by lifecycle state; 1/(s·$) is \
             speed per dollar (1 ÷ (p50 seconds × $/query)), higher is better."
                .to_string()
        },
        headers: if c.serving_groups.is_none() && !query_states.is_empty() {
            // Pivoted per-lifecycle-state table (vector): one row per state,
            // warm and cold sides together. FTS/SQL also carry lifecycle
            // states, but their serving rows come from the per-family groups
            // branch (7 columns) — the pivot headers apply only when the
            // per-state branch built the rows.
            vec![
                "State".into(),
                "recall@10".into(),
                "Payload".into(),
                "Egress $/1M".into(),
                "warm p50".into(),
                "Warm $/1M".into(),
                "cold 1st".into(),
                "cold steady".into(),
                "Cold steady $/1M".into(),
            ]
        } else {
            vec![
                "Query".into(),
                "p50".into(),
                "Payload".into(),
                "Egress $/1M".into(),
                "queries/$".into(),
                "1/(s·$)".into(),
                "$/1M (total)".into(),
            ]
        },
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
    // Warm and cold read lines are rate references (empty $/month) when both
    // sides were measured — the blended line is the billed monthly read cost
    // then. When only one side was measured, that side's own row carries the
    // billed figure instead (see the row-emission match below), so this must
    // stay in lock-step with every row that fills in a $/month cell. Each
    // per-query price already carries its RAM-hold leg for the full resident
    // set.
    let blended_read_q = match (&steady_warm, &steady_cold) {
        (Some((_, warm_q)), Some((_, cold_q))) => {
            Some(warm_q * summary_warm_fraction() + cold_q * (1.0 - summary_warm_fraction()))
        }
        (Some((_, warm_q)), None) => Some(*warm_q),
        (None, Some((_, cold_q))) => Some(*cold_q),
        (None, None) => None,
    };
    // Every dollar `blended_read_q` contributes to the Total must appear on a
    // rendered row: with both warm and cold measured, the individual rows
    // are reference-only (blank $/month) and the blend below carries the
    // billed figure; with only one side measured, that side's own row IS
    // the billed line, so its $/month cell is filled instead of left blank.
    match (&steady_warm, &steady_cold) {
        (Some((label, per_q)), Some(_)) => {
            summary_rows.push(vec![
                text(format!(
                    "Reads — {} queries/mo, {label}",
                    fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
                )),
                text(usd_per_million(*per_q)),
                text(""),
            ]);
        }
        (Some((label, per_q)), None) => {
            let month = per_q * SUMMARY_QUERIES_PER_MONTH;
            summary_rows.push(vec![
                text(format!(
                    "Reads — {} queries/mo, {label} (100% warm — no cold measured)",
                    fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
                )),
                text(usd_per_million(*per_q)),
                metric(month, usd(month), Better::Lower),
            ]);
        }
        (None, _) => {}
    }
    match (&steady_warm, &steady_cold) {
        (Some(_), Some((label, per_q))) => {
            summary_rows.push(vec![
                text(format!(
                    "Reads — {} queries/mo, {label}",
                    fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
                )),
                text(usd_per_million(*per_q)),
                text(""),
            ]);
        }
        (None, Some((label, per_q))) => {
            let month = per_q * SUMMARY_QUERIES_PER_MONTH;
            summary_rows.push(vec![
                text(format!(
                    "Reads — {} queries/mo, {label} (100% cold — no warm measured)",
                    fmt_count(SUMMARY_QUERIES_PER_MONTH as usize)
                )),
                text(usd_per_million(*per_q)),
                metric(month, usd(month), Better::Lower),
            ]);
        }
        (_, None) => {}
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
                summary_warm_fraction() * 100.0,
                (1.0 - summary_warm_fraction()) * 100.0,
            )),
            text(usd_per_million(blended_q)),
            metric(month, usd(month), Better::Lower),
        ]);
    }
    // Writes priced at the corpus scale the bench actually measured — the
    // whole table written once per month — covering COMMITS only (ingest +
    // the delta commit). The hidden-index BUILD (drain) and the fold
    // (optimize) are the dominant make-queryable cost but are billed on the
    // Maintenance lines below, so this line is commit compute only and the
    // total never double-counts them. Read the make-searchable cost as
    // Writes + Maintenance, not Writes alone.
    let writes_month = ingest_compute.unwrap_or(0.0)
        + ingest_req_usd
        + delta_compute.unwrap_or(0.0)
        + delta_req_usd;
    summary_rows.push(vec![
        text(format!(
            "Writes — {} docs/mo (commit compute only; drain/optimize in Maintenance)",
            fmt_count(c.n_docs)
        )),
        text(if write_compute_metered {
            format!("{}/1M docs", usd(per_million_docs(writes_month)))
        } else {
            format!(
                "{}/1M docs (compute NOT METERED)",
                usd(per_million_docs(writes_month))
            )
        }),
        metric(writes_month, usd(writes_month), Better::Lower),
    ]);
    // Maintenance: per-event rates for open / drain / compaction, then one
    // billed line at the steady-state cadence — the whole corpus is drained
    // and fully optimized once per month each (both `drain_pass_usd` and
    // `compact_pass_usd` are whole-corpus passes), one table open per pass.
    // Compaction tracks the write/drain cadence because the summary prices
    // reads at the post-compact query latencies, which a rarer optimize could
    // not sustain. The compaction-only (text) path has no drain and just
    // optimizes the whole corpus once per month.
    let drain_pass_usd = has_drain.then(|| drain_compute.unwrap_or(0.0) + drain_req_usd);
    let compact_pass_usd =
        has_compaction.then(|| compaction_compute.unwrap_or(0.0) + compaction_req_usd);
    // A maintenance phase that RAN but whose CPU was not sampled contributes
    // only its request leg; the pass (and the monthly maintenance line built
    // from it) must say so rather than let the missing compute read as free.
    let drain_compute_metered = !has_drain || drain_compute.is_some();
    let compact_compute_metered = !has_compaction || compaction_compute.is_some();
    let maintenance_metered = drain_compute_metered && compact_compute_metered;
    let not_metered_suffix = |metered: bool| {
        if metered {
            ""
        } else {
            " (compute NOT METERED)"
        }
    };
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
                let compute = q
                    .open_cpu_s
                    .map(|cpu| {
                        let billed = cpu.max(inst.ram_leg(q.open_s, Some(c.resident_anon_bytes)));
                        inst.compute_usd(billed)
                    })
                    .unwrap_or(0.0);
                compute + c.store.cold_open.map(|io| request_usd(&io)).unwrap_or(0.0)
            })
        });
    // Background lazy->mmap cache fill, concurrent with the cold/repeat
    // windows (the I/O ledger's "Fill" row prices the identical quantity —
    // see per_query_row's Fill construction above, which this mirrors). It
    // is a ONE-TIME cost per cold-open cycle (the fill runs once, warming
    // the cache for whatever queries follow), not a recurring per-query
    // cost — folding it into the recurring cold rate would bill every
    // steady cold query for a fill that happened once. Shown as its own
    // line, like the bulk per-event rates, rather than silently dropped.
    let fill_meter =
        query_states
            .last()
            .and_then(|state| match (state.io.cold_query, state.io.cold_repeat) {
                (Some(q), Some(r)) => Some(merge_background_fill(&q, &r)),
                (Some(q), None) => Some(background_fill_meter(&q)),
                (None, Some(r)) => Some(background_fill_meter(&r)),
                (None, None) => None,
            });
    if let Some(fill) = fill_meter.filter(|f| f.get_count > 0) {
        let fill_usd = request_usd(&fill);
        summary_rows.push(vec![
            text(format!(
                "Cache fill (one-time, per cold open, not in Total) — {} GET",
                fill.get_count
            )),
            text(format!("{}/1M queries", usd(fill_usd * PER_MILLION))),
            text("—"),
        ]);
    }
    if let Some(open) = steady_open_usd {
        summary_rows.push(vec![
            text("Open — cold table open"),
            text(format!("{}/open", usd(open))),
            text(""),
        ]);
    }
    if let Some(drain) = drain_pass_usd {
        summary_rows.push(vec![
            text("Drain — one full-corpus pass"),
            text(format!(
                "{}/pass{}",
                usd(drain),
                not_metered_suffix(drain_compute_metered)
            )),
            text(""),
        ]);
    }
    if let Some(compact) = compact_pass_usd {
        summary_rows.push(vec![
            text("Compaction — one optimize pass"),
            text(format!(
                "{}/pass{}",
                usd(compact),
                not_metered_suffix(compact_compute_metered)
            )),
            text(""),
        ]);
    }
    let maintenance_month = if let Some(drain) = drain_pass_usd {
        // Steady state on the Writes basis (the whole table is (re)written, and
        // therefore drained, once per month): `drain_pass_usd` already measures
        // ONE whole-corpus drain, so it is billed once per month. The old
        // weighting (n_commits / 16) treated the whole-corpus drain as a
        // 16-commit pass, which only equalled 1.0 at n_commits == 16 (i.e.
        // <= 50M, where n_commits is pinned at the 16-commit floor) and
        // double-counted the drain past that (2x at 100M, growing with scale).
        // Normalizing to the corpus is scale-stable and identical at <= 50M.
        let drains_mo = 1.0;
        // Compaction tracks the drain/write cadence. The summary prices reads
        // at the POST-COMPACT (fully-optimized) query latencies, and those are
        // only sustainable if the table is re-optimized about as often as it is
        // drained — a rarer horizon would leave un-merged deltas accumulating
        // and degrade the very latencies being priced. So a full optimize runs
        // once per month too, at the same weight as the drain.
        let compacts_mo = drains_mo;
        let opens_mo = drains_mo + compacts_mo;
        let month = drains_mo * drain
            + compacts_mo * compact_pass_usd.unwrap_or(0.0)
            + opens_mo * steady_open_usd.unwrap_or(0.0);
        summary_rows.push(vec![
            text("Maintenance — drain + full-corpus optimize, 1×/mo each"),
            text(format!(
                "{} drains + {} compactions + {} opens/mo{}",
                fmt_events(drains_mo),
                fmt_events(compacts_mo),
                fmt_events(opens_mo),
                not_metered_suffix(maintenance_metered),
            )),
            metric(month, usd(month), Better::Lower),
        ]);
        Some(month)
    } else if let Some(compact) = compact_pass_usd {
        // Compaction-only (text) cadence: no drain, so the whole corpus is
        // fully optimized once per monthly write cycle — same rationale as the
        // drain-based branch (sustains the post-compact query latencies the
        // summary prices) and corpus-normalized (scale-stable, vs the old
        // n_commits / 256 form that over-counted the whole-corpus optimize 2x
        // at 100M), one open per pass.
        let compacts_mo = 1.0;
        let opens_mo = compacts_mo;
        let month = compacts_mo * compact + opens_mo * steady_open_usd.unwrap_or(0.0);
        summary_rows.push(vec![
            text("Maintenance — full-corpus optimize, 1×/mo"),
            text(format!(
                "{} compactions + {} opens/mo{}",
                fmt_events(compacts_mo),
                fmt_events(opens_mo),
                not_metered_suffix(compact_compute_metered),
            )),
            metric(month, usd(month), Better::Lower),
        ]);
        Some(month)
    } else {
        None
    };
    // Egress: the result payload leaving the engine to the client, priced on a
    // SEPARATE line from reads (compute + requests) — a different network hop
    // (engine→client, not S3→engine), so it never double-counts the request or
    // compute legs. Payload is cache-independent, so the warm battery-mean
    // payload is charged on every served query; `payload_bytes` on each warm
    // shape is the logical size of the values returned.
    // Egress on the served queries. BOUNDED-result shapes only: their payload
    // is O(k), so a mean is a real per-query rate. Unbounded (bulk) shapes are
    // excluded and billed per event below — averaging a full-corpus dump into
    // a per-query rate would assert that every query returns one, which is how
    // a battery of mostly-KB results turns into a multi-hundred-dollar line.
    let egress_month = {
        let payloads: Vec<u64> = if !monthly_bounded_payloads.is_empty() {
            monthly_bounded_payloads.clone()
        } else {
            c.warm.iter().map(|w| w.payload_bytes).collect()
        };
        if payloads.is_empty() {
            None
        } else {
            let mean_bytes = payloads.iter().sum::<u64>() as f64 / payloads.len() as f64;
            let month = egress_usd(mean_bytes) * SUMMARY_QUERIES_PER_MONTH;
            summary_rows.push(vec![
                text(format!(
                    "Egress — {} queries/mo × {} payload (mean of {} bounded-result shapes)",
                    fmt_count(SUMMARY_QUERIES_PER_MONTH as usize),
                    fmt_bytes(mean_bytes as u64),
                    payloads.len(),
                )),
                text(format!(
                    "{}/1M queries",
                    usd(egress_usd(mean_bytes) * PER_MILLION)
                )),
                metric(month, usd(month), Better::Lower),
            ]);
            Some(month)
        }
    };
    // Bulk row sets and scan-backed aggregates as per-event rates: the cost
    // of ONE such query, which the reader multiplies by however many they
    // actually run. Not folded into the monthly total — the run rate is a
    // workload property, not a bench constant.
    for (name, warm_usd, payload) in &monthly_bulk_shapes {
        let per_event = warm_usd + egress_usd(*payload as f64);
        summary_rows.push(vec![
            text(format!(
                "Per event (not in Total) — {name} ({} result)",
                fmt_bytes(*payload)
            )),
            text(format!("{}/query", usd(per_event))),
            text("—"),
        ]);
    }
    let monthly_total = storage_month
        + blended_read_q
            .map(|q| q * SUMMARY_QUERIES_PER_MONTH)
            .unwrap_or(0.0)
        + writes_month
        + maintenance_month.unwrap_or(0.0)
        + egress_month.unwrap_or(0.0);
    summary_rows.push(vec![
        text("Total (storage + blended reads + writes + maintenance + egress)"),
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

    // ---- Block 6: provisioned occupancy (informational) ----
    // The keep-warm framing the marginal summary above deliberately excludes:
    // what holding this tenant's capacity costs when it is RESERVED between
    // queries rather than billed per query served. Models a process-per-
    // database serving platform with scale-to-zero — an idle tenant keeps
    // only its local NVMe cache (worker reaped: no RAM, no CPU); an active
    // tenant's node share is bounded by its largest resource share. Never
    // added to the marginal Total: the two framings answer different
    // questions (cost of work performed vs cost of capacity held), and which
    // applies is the operator's keep-warm policy. Every row is measured — an
    // unmeasured resource is omitted, never guessed at 0.
    let replicas = occupancy_replicas();
    let node_month = inst.usd_per_month();
    // Same resident-set basis the serving table and the per-query RAM-hold
    // leg use: the LAST populated query state (post-compact when the
    // lifecycle ran) is the shape a long-lived table serves.
    let occupancy_resident = query_states
        .last()
        .map(|s| s.serving_resident_bytes(c.resident_anon_bytes))
        .unwrap_or(c.resident_anon_bytes);
    let occupancy_ram_label = query_states
        .last()
        .map(|s| s.serving_ram_label(c.resident_anon_bytes))
        .unwrap_or_else(|| fmt_bytes(occupancy_resident));
    // Billed compute per query at the same 95/5 blend (and the same
    // single-sided fallback) as the monthly read line.
    let blended_vcpu_q = match (steady_warm_vcpu_s, steady_cold_vcpu_s) {
        (Some(warm_v), Some(cold_v)) => {
            Some(warm_v * summary_warm_fraction() + cold_v * (1.0 - summary_warm_fraction()))
        }
        (Some(warm_v), None) => Some(warm_v),
        (None, Some(cold_v)) => Some(cold_v),
        (None, None) => None,
    };
    let cpu_sh = blended_vcpu_q.map(|v| inst.cpu_share_at_load(v, SUMMARY_QUERIES_PER_MONTH));
    let ram_sh = (occupancy_resident > 0).then(|| inst.ram_share(occupancy_resident));
    let nvme_sh = c.store.disk_cache_bytes.map(|b| inst.nvme_share(b));
    let mut occupancy_rows: Vec<Vec<Cell>> = Vec::new();
    if let (Some(share), Some(vcpu_q)) = (cpu_sh, blended_vcpu_q) {
        occupancy_rows.push(vec![
            text("CPU at assumed load"),
            text(format!(
                "{} queries/mo × {} billed vCPU·s/query on {} vCPU",
                fmt_count(SUMMARY_QUERIES_PER_MONTH as usize),
                fmt_vcpu_seconds(vcpu_q),
                inst.vcpu,
            )),
            context(share, fmt_share(share), Better::Lower),
            text(""),
            text(""),
        ]);
    }
    if let Some(share) = ram_sh {
        occupancy_rows.push(vec![
            text("RAM-resident (while worker live)"),
            text(format!(
                "{occupancy_ram_label} of {:.0} GiB — page cache is node-global and \
                 reclaimable, not a standing per-tenant reservation",
                inst.ram_gib,
            )),
            context(share, fmt_share(share), Better::Lower),
            text(""),
            text(""),
        ]);
    }
    if let (Some(share), Some(bytes)) = (nvme_sh, c.store.disk_cache_bytes) {
        occupancy_rows.push(vec![
            text("NVMe disk cache"),
            text(format!(
                "{} local cache (user + hidden index, shared root) of {:.0} GB",
                fmt_bytes(bytes),
                inst.nvme_gb,
            )),
            context(share, fmt_share(share), Better::Lower),
            text(""),
            text(""),
        ]);
    }
    let active_binding = binding_share(&[("CPU", cpu_sh), ("RAM", ram_sh), ("NVMe", nvme_sh)]);
    if let Some((binding, share)) = active_binding {
        let per_replica = share * node_month;
        let total = per_replica * f64::from(replicas);
        occupancy_rows.push(vec![
            text(format!("Active occupancy (binding: {binding})")),
            text(format!(
                "largest share × {}/node-mo, held while the worker process is live",
                usd(node_month),
            )),
            context(share, fmt_share(share), Better::Lower),
            context(per_replica, usd(per_replica), Better::Lower),
            context(total, usd(total), Better::Lower),
        ]);
    }
    if let Some(share) = nvme_sh {
        let per_replica = share * node_month;
        let total = per_replica * f64::from(replicas);
        occupancy_rows.push(vec![
            text("Idle-retained occupancy (NVMe only)"),
            text("cache kept on local disk between activations; worker reaped — zero RAM/CPU"),
            context(share, fmt_share(share), Better::Lower),
            context(per_replica, usd(per_replica), Better::Lower),
            context(total, usd(total), Better::Lower),
        ]);
    }
    let occupancy = (!occupancy_rows.is_empty()).then(|| Block {
        subtitle: format!(
            "Provisioned occupancy — keep-warm framing (informational, NOT in the Total \
             above): what holding this tenant's capacity costs when it is reserved between \
             queries instead of billed per query served. Share of one {} ({} vCPU / {:.0} GiB \
             RAM / {:.0} GB NVMe at ${:.4}/h = {}/node-mo) × R={replicas} replicas (R-way HA, \
             each replica keeps an independent local cache; INFINO_BENCH_COST_REPLICAS \
             overrides). Shares are of raw instance capacity — apply any usable-capacity \
             headroom policy as your own divisor.",
            inst.name,
            inst.vcpu,
            inst.ram_gib,
            inst.nvme_gb,
            inst.usd_per_hour,
            usd(node_month),
        ),
        headers: vec![
            "Line".into(),
            "Measured basis".into(),
            "Share of node".into(),
            "$/mo — 1 replica".into(),
            format!("$/mo — ×{replicas} replicas"),
        ],
        rows: occupancy_rows,
    });

    // ---- Block 7: serving COGS per keep-warm policy ----
    // The number the marginal summary and the occupancy view each show one
    // half of: what serving this tenant costs per month, one row per
    // keep-warm policy. Egress is excluded throughout — passed through to
    // the customer at cost, revenue-neutral, not COGS. These rows are the
    // COGS basis for pricing; the assumptions are the knobs printed in the
    // subtitle, all env-overridable, so scenarios re-run without a code
    // change. Unmeasured components withhold their row, never guess.
    let warm_frac = summary_warm_fraction();
    let marginal_ex_egress = storage_month
        + blended_read_q
            .map(|q| q * SUMMARY_QUERIES_PER_MONTH)
            .unwrap_or(0.0)
        + writes_month
        + maintenance_month.unwrap_or(0.0);
    let idle_floor_total = nvme_sh.map(|s| s * node_month * f64::from(replicas));
    let active_floor_total = active_binding.map(|(_, s)| s * node_month * f64::from(replicas));
    let warm_reads_month = steady_warm
        .as_ref()
        .map(|(_, q)| q * SUMMARY_QUERIES_PER_MONTH);
    let serving_cogs = {
        let warm_pct = warm_frac * 100.0;
        let cold_pct = 100.0 - warm_pct;
        let mut rows = vec![vec![
            text("Policy A — scale-to-zero (nothing retained between queries)"),
            text(format!(
                "storage {} + reads {} ({warm_pct:.0}% warm / {cold_pct:.0}% at Azure-cold) \
                 + writes {} + maintenance {}",
                usd(storage_month),
                usd(blended_read_q
                    .map(|q| q * SUMMARY_QUERIES_PER_MONTH)
                    .unwrap_or(0.0)),
                usd(writes_month),
                usd(maintenance_month.unwrap_or(0.0)),
            )),
            metric(marginal_ex_egress, usd(marginal_ex_egress), Better::Lower),
        ]];
        match serving_cogs_month(marginal_ex_egress, idle_floor_total) {
            Some(total) => rows.push(vec![
                text("Policy B — keep-warm NVMe (worker reaped between queries)"),
                text(format!(
                    "policy A {} + idle-NVMe floor {} (×R={replicas}). Conservative: the \
                     {cold_pct:.0}% misses are still priced at Azure-cold, but with NVMe \
                     retained a miss costs near the warm rate — true B is at most this",
                    usd(marginal_ex_egress),
                    usd(idle_floor_total.unwrap_or(0.0)),
                )),
                metric(total, usd(total), Better::Lower),
            ]),
            None => rows.push(vec![
                text("Policy B — keep-warm NVMe (worker reaped between queries)"),
                text("NVMe cache unmeasured this run — row withheld, never guessed"),
                text("—"),
            ]),
        }
        match (active_floor_total, warm_reads_month) {
            (Some(active), Some(warm_reads)) => {
                let total = storage_month
                    + warm_reads
                    + writes_month
                    + maintenance_month.unwrap_or(0.0)
                    + active;
                rows.push(vec![
                    text("Policy C — reserved (worker held live, 100% warm)"),
                    text(format!(
                        "storage {} + reads at 100% warm {} + writes {} + maintenance {} + \
                         active occupancy {} (binding share × node-mo × R={replicas})",
                        usd(storage_month),
                        usd(warm_reads),
                        usd(writes_month),
                        usd(maintenance_month.unwrap_or(0.0)),
                        usd(active),
                    )),
                    metric(total, usd(total), Better::Lower),
                ]);
            }
            _ => rows.push(vec![
                text("Policy C — reserved (worker held live, 100% warm)"),
                text("needs a measured warm read class and an occupancy share — row withheld"),
                text("—"),
            ]),
        }
        Block {
            subtitle: format!(
                "Serving COGS per keep-warm policy — one tenant-month at {} queries/mo, \
                 egress excluded (passed through at cost). Pick the policy you sell; \
                 assumptions are knobs: warm-hit rate {warm_pct:.0}% \
                 (INFINO_BENCH_COST_WARM_FRACTION), R={replicas} \
                 (INFINO_BENCH_COST_REPLICAS), instance rates (INFINO_BENCH_COST_*).",
                fmt_count(SUMMARY_QUERIES_PER_MONTH as usize),
            ),
            headers: vec!["Line".into(), "Basis".into(), "$/month".into()],
            rows,
        }
    };

    let mut blocks = vec![rate_card];
    if let Some(io_ledger) = io_ledger {
        blocks.push(io_ledger);
    }
    blocks.push(compute_ledger);
    blocks.push(serving);
    blocks.push(monthly_summary);
    if let Some(occupancy) = occupancy {
        blocks.push(occupancy);
    }
    blocks.push(serving_cogs);

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
/// Flatten FTS two-phase cold stats into cost rows: the search phase under
/// the shape's name and the fetch phase under "name (+fetch)" — matching the
/// warm entries `warm_from_fts` emits, so the serving groups price cold for
/// BOTH cost classes.
pub fn cold_from_fts_timings(
    cold: &HashMap<&'static str, crate::executors::fts::FtsColdStat>,
) -> Vec<ColdQuery> {
    let one = |name: String, t: &ColdTiming| ColdQuery {
        name,
        open_s: t.open.as_secs_f64(),
        search_s: t.search.as_secs_f64(),
        open_cpu_s: t.open_cpu_s,
        search_cpu_s: t.search_cpu_s,
        search_get_count: t.search_get_count,
        search_get_bytes: t.search_get_bytes,
    };
    let mut out: Vec<ColdQuery> = cold
        .iter()
        .flat_map(|(name, s)| {
            std::iter::once(one((*name).to_string(), &s.search)).chain(
                s.fetched
                    .as_ref()
                    .map(|f| one(format!("{name}{FTS_FETCH_SUFFIX}"), f)),
            )
        })
        .collect();
    // Source is a HashMap: sort so row order — and any positional fallback
    // that picks a representative shape — is stable across runs.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub fn cold_from_timings(cold: &HashMap<&'static str, ColdTiming>) -> Vec<ColdQuery> {
    let mut out: Vec<ColdQuery> = cold
        .iter()
        .map(|(name, t)| ColdQuery {
            name: (*name).to_string(),
            open_s: t.open.as_secs_f64(),
            search_s: t.search.as_secs_f64(),
            open_cpu_s: t.open_cpu_s,
            search_cpu_s: t.search_cpu_s,
            search_get_count: t.search_get_count,
            search_get_bytes: t.search_get_bytes,
        })
        .collect();
    // Source is a HashMap: sort for run-to-run stable ordering.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Flatten warm FTS stats into `(name, p50_seconds, measured_cpu_seconds)`.
/// Suffix distinguishing an FTS shape's fetch-phase (retrieval) entry from its
/// query-phase (search) entry in the warm cost battery and serving groups.
pub const FTS_FETCH_SUFFIX: &str = " (+fetch)";

/// Two entries per shape — the two FTS cost classes: the query phase
/// (search: id + score, its own p50/CPU/payload) and the fetch phase
/// (retrieval: + top-k text, its own p50/CPU/payload) — so each class row in
/// the serving table carries internally consistent latency, compute, and
/// egress instead of pairing the search p50 with the fetched payload.
pub fn warm_from_fts(stats: &[FtsQueryStat]) -> Vec<WarmQueryCost> {
    stats
        .iter()
        .flat_map(|s| {
            [
                WarmQueryCost {
                    name: s.name.to_string(),
                    p50_s: s.warm.p50.as_secs_f64(),
                    cpu_s: s.cpu_s,
                    payload_rows: s.search_payload_rows,
                    payload_bytes: s.search_payload_bytes,
                },
                WarmQueryCost {
                    name: format!("{}{FTS_FETCH_SUFFIX}", s.name),
                    p50_s: s.fetched.p50.as_secs_f64(),
                    cpu_s: s.fetched_cpu_s,
                    payload_rows: s.fetched_payload_rows,
                    payload_bytes: s.fetched_payload_bytes,
                },
            ]
        })
        .collect()
}

/// Flatten warm SQL query sets into `(name, p50_seconds, measured_cpu_seconds)`.
pub fn warm_from_sql(sets: &QuerySets) -> Vec<WarmQueryCost> {
    sets.scalar
        .iter()
        .chain(&sets.tvf)
        .chain(&sets.fts_pushdown)
        .chain(&sets.agg_idx)
        .chain(&sets.agg_scan)
        .map(|s| WarmQueryCost {
            name: s.name.to_string(),
            p50_s: s.warm.p50.as_secs_f64(),
            cpu_s: s.cpu_s,
            payload_rows: s.rows as u64,
            payload_bytes: s.payload_bytes,
        })
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
                WarmQueryCost {
                    name: label,
                    p50_s: w.warm.p50.as_secs_f64(),
                    cpu_s: w.cpu_s,
                    // Vector's realistic result is id + score (no scalar decode);
                    // that is the payload measured on the warm path.
                    payload_rows: w.payload_rows,
                    payload_bytes: w.payload_bytes,
                }
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
                    search_get_count: t.search_get_count,
                    search_get_bytes: t.search_get_bytes,
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

    #[test]
    fn nvme_share_is_decimal_gb_fraction() {
        let inst = test_instance();
        // NVMe capacity is quoted in decimal GB, so 47.4e9 bytes on a 237 GB
        // device is exactly a 20% share — a GiB divisor would misstate it.
        assert!((inst.nvme_share(47_400_000_000) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cpu_share_at_load_reconciles_with_month_seconds() {
        let inst = test_instance();
        // An 8-vCPU node holds 8 × 730.5 h × 3600 s = 21,038,400 vCPU·s per
        // month; a load consuming exactly that many is a share of 1.0, and
        // half the load is half the share.
        let node_vcpu_s = 8.0 * HOURS_PER_MONTH * SECS_PER_HOUR;
        assert!((inst.cpu_share_at_load(node_vcpu_s / 1e6, 1e6) - 1.0).abs() < 1e-12);
        assert!((inst.cpu_share_at_load(node_vcpu_s / 2e6, 1e6) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn binding_share_picks_largest_measured_and_skips_unmeasured() {
        let picked = binding_share(&[("CPU", Some(0.1)), ("RAM", Some(0.6)), ("NVMe", Some(0.2))]);
        assert_eq!(picked, Some(("RAM", 0.6)));
        // An unmeasured share is skipped, never treated as a 0 that "loses"
        // the max — with RAM absent, NVMe binds.
        let picked = binding_share(&[("CPU", Some(0.1)), ("RAM", None), ("NVMe", Some(0.2))]);
        assert_eq!(picked, Some(("NVMe", 0.2)));
        // Nothing measured ⇒ no binding row, not a fabricated $0 one.
        assert_eq!(binding_share(&[("CPU", None), ("RAM", None)]), None);
    }

    #[test]
    fn occupancy_dollars_multiply_share_node_price_and_replicas() {
        let inst = test_instance();
        // usd_per_month is the plain hourly rate over the calendar month, and
        // a 50% share on 2 replicas bills exactly one full node-month.
        assert!((inst.usd_per_month() - 0.3629 * HOURS_PER_MONTH).abs() < 1e-9);
        let billed = 0.5 * inst.usd_per_month() * 2.0;
        assert!((billed - inst.usd_per_month()).abs() < 1e-9);
    }

    /// The warm-fraction knob accepts only a real hit rate: 0 < f ≤ 1.
    /// Unset, garbage, zero, negative, >1 and NaN all fall back to the
    /// default rather than silently shaping the blend.
    #[test]
    fn warm_fraction_parse_accepts_only_a_real_hit_rate() {
        assert_eq!(
            parse_warm_fraction(None),
            DEFAULT_SUMMARY_READ_WARM_FRACTION
        );
        assert_eq!(parse_warm_fraction(Some("0.5")), 0.5);
        assert_eq!(parse_warm_fraction(Some("1")), 1.0);
        for bad in ["0", "-0.2", "1.5", "NaN", "warm"] {
            assert_eq!(
                parse_warm_fraction(Some(bad)),
                DEFAULT_SUMMARY_READ_WARM_FRACTION,
                "{bad} must fall back"
            );
        }
    }

    /// The composed serving-COGS number: marginal-ex-egress plus the
    /// keep-warm floor, and withheld (not guessed at marginal-only) when
    /// the floor is unmeasured. Values mirror the 100K FTS reference run:
    /// $4.18 marginal + $0.78 idle-NVMe floor = $4.96/mo.
    #[test]
    fn serving_cogs_composes_marginal_plus_floor_or_withholds() {
        let composed = serving_cogs_month(4.18, Some(0.78)).expect("floor measured");
        assert!((composed - 4.96).abs() < 1e-9);
        assert_eq!(serving_cogs_month(4.18, None), None);
    }

    #[test]
    fn replicas_parse_defaults_and_rejects_zero() {
        // Pure parser (no env mutation): unset and garbage fall back to the
        // default, and R=0 is rejected — it would print a $0 keep-warm bill,
        // which is never a measurement.
        assert_eq!(parse_replicas(None), DEFAULT_OCCUPANCY_REPLICAS);
        assert_eq!(parse_replicas(Some("1")), 1);
        assert_eq!(parse_replicas(Some("3")), 3);
        assert_eq!(parse_replicas(Some("0")), DEFAULT_OCCUPANCY_REPLICAS);
        assert_eq!(parse_replicas(Some("x")), DEFAULT_OCCUPANCY_REPLICAS);
    }

    #[test]
    fn fmt_share_keeps_tiny_shares_visible() {
        // The CPU share at 1M queries/mo is often well below 0.1% — it must
        // print with significant digits, never collapse to "0.0%".
        assert_eq!(fmt_share(0.593), "59.3%");
        assert_eq!(fmt_share(0.0006), "0.060%");
        assert_eq!(fmt_share(0.0), "0%");
    }
}
