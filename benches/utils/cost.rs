// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cost model for the bench — turns measured latency / footprint into
//! dollars, per the rule "a resource costs money only to the extent that
//! holding it blocks the next tenant."
//!
//! Three buckets, kept separate:
//!
//!   1. **Compute (instance-time).** Priced at the instance's marginal
//!      rate on the *binding* resource. Ingest saturates cores → CPU-time
//!      binds. Serving charges one core for `p50`, or resident anonymous
//!      heap if that is tighter than `1/vCPU`.
//!   2. **Object-store requests.** PUTs on ingest (counted). Cold GET/HEAD
//!      are priced only when `CellCost::cold_store` is populated from a
//!      metered cold iteration — otherwise cold object-store dollars are
//!      omitted, not guessed.
//!   3. **Object-store capacity.** `stored_GB · $/GB-month`.
//!
//! Local NVMe (file-backed disk-cache mmap) is treated as free.

use std::sync::OnceLock;

use crate::report::{Better, Block, Report, Section, metric, text};
use crate::storage_meter::ObjectStoreMeter;

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

    fn ingest_compute_usd(&self, wall_s: f64, writers: u32) -> f64 {
        self.ingest_vcpu_seconds(wall_s, writers) * self.usd_per_sec()
    }

    fn ingest_vcpu_seconds(&self, wall_s: f64, writers: u32) -> f64 {
        let cpu_share = f64::from(writers.min(self.vcpu)) / f64::from(self.vcpu.max(1));
        wall_s * cpu_share
    }

    fn per_query_usd(&self, p50_s: f64, resident_anon_bytes: u64) -> f64 {
        self.per_query_vcpu_seconds(p50_s, resident_anon_bytes) * self.usd_per_sec()
    }

    fn per_query_vcpu_seconds(&self, p50_s: f64, resident_anon_bytes: u64) -> f64 {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        p50_s * cpu_share.max(ram_share)
    }

    fn ram_binds(&self, resident_anon_bytes: u64) -> bool {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        ram_share > cpu_share
    }
}

/// Cold open + search latency for one query shape.
pub struct ColdQuery {
    pub name: String,
    pub open_s: f64,
    pub search_s: f64,
}

/// Everything one cell (one tier × modality) needs to be priced.
pub struct CellCost<'a> {
    pub ingest_wall_s: f64,
    pub writers: u32,
    pub put_count: u64,
    pub stored_bytes: u64,
    pub corpus_bytes: u64,
    pub n_docs: usize,
    pub resident_anon_bytes: u64,
    pub warm: &'a [(String, f64)],
    /// Cold latency rows (open and search timed separately).
    pub cold: Option<&'a [ColdQuery]>,
    /// Object-store HEAD/GET counts from one metered cold iteration (supertable only).
    pub cold_store: Option<ObjectStoreMeter>,
    /// Assumed retention for the capacity line (GB-months). Default 1 month.
    pub storage_months: Option<f64>,
    /// Whether a cold `open` is a one-time table/namespace open that is
    /// amortized across every query (supertable: manifest load + consumer
    /// setup, paid once), rather than per-query latency. For a single
    /// superfile the open is part of each cold read, so this is `false`.
    pub cold_open_amortized: bool,
}

fn usd(v: f64) -> String {
    if v < 0.01 {
        format!("${:.4}", v)
    } else {
        format!("${:.2}", v)
    }
}

fn usd_per_gb(v: f64) -> String {
    if v < 0.01 {
        format!("${:.4}/GB", v)
    } else {
        format!("${:.2}/GB", v)
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
        format!("{s:.1} vCPU·s")
    } else {
        format!("{s:.2} vCPU·s")
    }
}

fn fmt_wall_seconds(s: f64) -> String {
    if s >= 10.0 {
        format!("{s:.1}s wall")
    } else {
        format!("{s:.2}s wall")
    }
}

pub fn emit(report: &mut Report, anchor: &str, title: String, c: &CellCost) {
    let inst = Instance::current();
    let retention_months = c.storage_months.unwrap_or_else(storage_months);

    let ingest_vcpu_s = inst.ingest_vcpu_seconds(c.ingest_wall_s, c.writers);
    let compute = inst.ingest_compute_usd(c.ingest_wall_s, c.writers);
    let requests = c.put_count as f64 * USD_PER_PUT;
    let ingest_total = compute + requests;
    let per_million = if c.n_docs > 0 {
        ingest_total / (c.n_docs as f64 / 1.0e6)
    } else {
        0.0
    };
    let stored_gb = c.stored_bytes as f64 / BYTES_PER_GB;
    let gb_months = stored_gb * retention_months;
    let storage_month = gb_months * USD_PER_GB_MONTH;
    let storage_per_million_docs_month = if c.n_docs > 0 {
        storage_month / (c.n_docs as f64 / 1.0e6)
    } else {
        0.0
    };
    let write_rate_per_gb = if stored_gb > 0.0 {
        ingest_total / stored_gb
    } else {
        0.0
    };

    let warm_costs: Vec<(f64, f64, String)> = c
        .warm
        .iter()
        .map(|(name, p50_s)| {
            let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
            (per_q, *p50_s, name.clone())
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
    let min_per_million_q = min_q_cost * 1.0e6;
    let max_per_million_q = max_q_cost * 1.0e6;

    let anchor_cold = c.cold.and_then(|rows| {
        rows.iter()
            .find(|q| q.name == "ten_term_or")
            .or_else(|| rows.first())
    });

    let warm_query_cell = if warm_costs.is_empty() {
        "—".into()
    } else if (max_per_million_q - min_per_million_q).abs() < 1e-9 {
        format!(
            "{}/1M queries @ {} p50 ({})",
            usd(min_per_million_q),
            crate::markdown::fmt_time(fastest_p50 * 1e9),
            fastest_name,
        )
    } else {
        format!(
            "{}–{}/1M queries ({}–{} p50 battery)",
            usd(min_per_million_q),
            usd(max_per_million_q),
            crate::markdown::fmt_time(fastest_p50 * 1e9),
            crate::markdown::fmt_time(
                warm_costs
                    .iter()
                    .map(|(_, p50, _)| *p50)
                    .fold(0.0_f64, f64::max)
                    * 1e9,
            ),
        )
    };

    let mut rate_rows = vec![
        vec![
            text("Storage"),
            text(format!(
                "{} × {:.0} mo → {}/mo",
                usd_per_gb(USD_PER_GB_MONTH),
                retention_months,
                usd(storage_month),
            )),
        ],
        vec![
            text("Write once (ingest)"),
            text(format!(
                "{} → {} total",
                usd_per_gb(write_rate_per_gb),
                usd(ingest_total),
            )),
        ],
        vec![text("Warm query (marginal CPU)"), text(warm_query_cell)],
    ];

    // A supertable's cold `open` is the one-time table/namespace open
    // (manifest load + consumer setup), amortized across every query — it is
    // not per-query latency. Show it as its own one-time line and price the
    // cold *query* on `search` alone. A single superfile pays the open on
    // each cold read, so there the cold query is `open + search`.
    if let Some(q) = anchor_cold {
        if c.cold_open_amortized {
            rate_rows.push(vec![
                text("Table open (one-time, amortized)"),
                text(format!(
                    "{} — manifest + consumer, paid once per open",
                    crate::markdown::fmt_time(q.open_s * 1e9),
                )),
            ]);
            rate_rows.push(vec![
                text("Cold query (latency only — see search table)"),
                text(format!(
                    "{} search ({})",
                    crate::markdown::fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        } else {
            rate_rows.push(vec![
                text("Cold query (latency only — see search table)"),
                text(format!(
                    "{} open + {} search ({})",
                    crate::markdown::fmt_time(q.open_s * 1e9),
                    crate::markdown::fmt_time(q.search_s * 1e9),
                    q.name,
                )),
            ]);
        }
    }

    let rate_card = Block {
        subtitle: format!(
            "Rate card — {} docs, {} stored (Infino measured; latency lives in the \
             search table — warm vs cold are not interchangeable)",
            crate::markdown::fmt_count(c.n_docs),
            crate::rss::fmt_bytes(c.stored_bytes),
        ),
        headers: vec!["Line".into(), "Infino (measured)".into()],
        rows: rate_rows,
    };

    let writers_used = c.writers.min(inst.vcpu);
    let vcpu_share = format!("{writers_used}/{}/vCPU share", inst.vcpu);
    let ingest_cpu_row = vec![
        text(format!("Ingest CPU ({}w on {} vCPU)", c.writers, inst.vcpu)),
        text(format!(
            "{} = {} × {vcpu_share}",
            fmt_vcpu_seconds(ingest_vcpu_s),
            fmt_wall_seconds(c.ingest_wall_s),
        )),
        text(format!("@ ${:.4}/hr → {}", inst.usd_per_hour, usd(compute))),
        metric(compute, usd(compute), Better::Lower),
    ];
    let put_row = vec![
        text(format!("Ingest PUT ({} requests)", c.put_count)),
        text("once at commit"),
        text(format!("@ $5/1M → {}", usd(requests))),
        metric(requests, usd(requests), Better::Lower),
    ];
    let storage_row = vec![
        text(format!(
            "Stored capacity ({})",
            crate::rss::fmt_bytes(c.stored_bytes)
        )),
        text(format!(
            "{stored_gb:.2} GB × {retention_months:.0} mo = {gb_months:.2} GB·mo",
        )),
        text(format!("@ $0.023/GB·mo → {}/mo", usd(storage_month))),
        metric(
            storage_month,
            format!("{}/mo", usd(storage_month)),
            Better::Lower,
        ),
    ];

    let mut meter_rows = vec![ingest_cpu_row, put_row, storage_row];

    if let Some(q) = anchor_cold {
        let open_vcpu = inst.per_query_vcpu_seconds(q.open_s, c.resident_anon_bytes);
        let search_vcpu = inst.per_query_vcpu_seconds(q.search_s, c.resident_anon_bytes);
        let open_usd = open_vcpu * inst.usd_per_sec();
        let search_usd = search_vcpu * inst.usd_per_sec();
        let open_label = if c.cold_open_amortized {
            format!("Table open CPU (one-time, {})", q.name)
        } else {
            format!("Cold open CPU ({})", q.name)
        };
        meter_rows.push(vec![
            text(open_label),
            text(format!(
                "{} × {}",
                fmt_wall_seconds(q.open_s),
                fmt_vcpu_seconds(open_vcpu),
            )),
            text(format!(
                "@ ${:.4}/hr → {}",
                inst.usd_per_hour,
                usd(open_usd)
            )),
            metric(open_usd, usd(open_usd), Better::Lower),
        ]);
        meter_rows.push(vec![
            text(format!("Cold search CPU ({})", q.name)),
            text(format!(
                "{} × {}",
                fmt_wall_seconds(q.search_s),
                fmt_vcpu_seconds(search_vcpu),
            )),
            text(format!(
                "@ ${:.4}/hr → {}",
                inst.usd_per_hour,
                usd(search_usd)
            )),
            metric(search_usd, usd(search_usd), Better::Lower),
        ]);
    }

    if let Some(store) = c.cold_store {
        let head_usd = store.head_count as f64 * USD_PER_GET;
        let get_usd = store.get_count as f64 * USD_PER_GET;
        let req_usd = head_usd + get_usd;
        meter_rows.push(vec![
            text(format!(
                "Cold S3 HEAD ({} calls, one metered iter)",
                store.head_count
            )),
            text("one cold open + search"),
            text(format!("@ $0.40/1M → {}", usd(head_usd))),
            metric(head_usd, usd(head_usd), Better::Lower),
        ]);
        meter_rows.push(vec![
            text(format!(
                "Cold S3 GET ({} calls, {} fetched)",
                store.get_count,
                crate::rss::fmt_bytes(store.get_bytes),
            )),
            text("one cold open + search"),
            text(format!("@ $0.40/1M → {}", usd(get_usd))),
            metric(get_usd, usd(get_usd), Better::Lower),
        ]);
        meter_rows.push(vec![
            text("Cold object-store requests (total)"),
            text(format!(
                "{} HEAD + {} GET",
                store.head_count, store.get_count
            )),
            text(usd(req_usd)),
            metric(req_usd, usd(req_usd), Better::Lower),
        ]);
    } else if c.cold.is_some() {
        meter_rows.push(vec![
            text("Cold S3 HEAD/GET"),
            text("not metered in this cell yet"),
            text("cold object-store $ omitted"),
            text("—"),
        ]);
    }

    if let Some((name, p50_s)) = c
        .warm
        .iter()
        .find(|(n, _)| n == "ten_term_or")
        .or_else(|| c.warm.first())
    {
        let vcpu_s = inst.per_query_vcpu_seconds(*p50_s, c.resident_anon_bytes);
        let q_usd = vcpu_s * inst.usd_per_sec();
        meter_rows.push(vec![
            text(format!("Warm query CPU ({name})")),
            text(format!(
                "{} × {}",
                fmt_wall_seconds(*p50_s),
                fmt_vcpu_seconds(vcpu_s),
            )),
            text(format!("@ ${:.4}/hr → {}", inst.usd_per_hour, usd(q_usd))),
            metric(q_usd, usd(q_usd), Better::Lower),
        ]);
    }

    let resource_meter = Block {
        subtitle: format!(
            "Resource meter — quantities behind the dollars ({}, retention {:.0} mo assumed for storage)",
            inst.name, retention_months,
        ),
        headers: vec![
            "Resource".into(),
            "Quantity".into(),
            "Rate".into(),
            "Cost".into(),
        ],
        rows: meter_rows,
    };

    let ingest_storage = Block {
        subtitle: format!(
            "Ingest & storage — priced on {} ({} vCPU / {:.0} GiB / {:.0} GB NVMe @ ${:.4}/hr)",
            inst.name, inst.vcpu, inst.ram_gib, inst.nvme_gb, inst.usd_per_hour,
        ),
        headers: vec!["Component".into(), "Cost".into(), "Per-unit".into()],
        rows: vec![
            vec![
                text(format!(
                    "Ingest compute ({}w × {:.1}s)",
                    c.writers, c.ingest_wall_s
                )),
                metric(compute, usd(compute), Better::Lower),
                text(format!("{}/1M docs", usd(per_million))),
            ],
            vec![
                text(format!("Ingest requests (~{} PUT)", c.put_count)),
                metric(requests, usd(requests), Better::Lower),
                text(String::new()),
            ],
            vec![
                text(format!(
                    "Stored capacity ({})",
                    crate::rss::fmt_bytes(c.stored_bytes)
                )),
                metric(
                    storage_month,
                    format!("{}/mo", usd(storage_month)),
                    Better::Lower,
                ),
                text(format!(
                    "{}/1M docs·mo",
                    usd(storage_per_million_docs_month)
                )),
            ],
        ],
    };

    let binding = if inst.ram_binds(c.resident_anon_bytes) {
        "DRAM"
    } else {
        "CPU"
    };
    let serving_rows: Vec<Vec<_>> = c
        .warm
        .iter()
        .map(|(name, p50_s)| {
            let per_q = inst.per_query_usd(*p50_s, c.resident_anon_bytes);
            let per_q_usd = per_q.max(f64::MIN_POSITIVE);
            let queries_per_usd = 1.0 / per_q_usd;
            let per_million_q = per_q * 1.0e6;
            vec![
                text(name.clone()),
                text(crate::markdown::fmt_time(p50_s * 1.0e9)),
                metric(
                    queries_per_usd,
                    format!("{:.0}", queries_per_usd),
                    Better::Higher,
                ),
                text(usd(per_million_q)),
            ]
        })
        .collect();

    let serving = Block {
        subtitle: format!(
            "Serving — latency per dollar (binding: {binding}; resident heap {}, file-backed cache free on NVMe)",
            crate::rss::fmt_bytes(c.resident_anon_bytes),
        ),
        headers: vec![
            "Query".into(),
            "p50".into(),
            "queries/$".into(),
            "$/1M queries".into(),
        ],
        rows: serving_rows,
    };

    report.emit(&Section {
        anchor: anchor.into(),
        title,
        note: "Cost model on measured bench rows. The **Resource meter** lists vCPU·seconds, \
               PUT counts, stored GB·months, and wall times. Cold S3 HEAD/GET counts and \
               dollars appear only when a metered cold iteration ran (`cold_store` set). \
               For a supertable the cold `open` is the one-time table open (manifest + \
               consumer), amortized across queries — it is a separate line, not per-query \
               latency. Warm vs cold latency is in the search table. Δ is vs the previous run."
            .into(),
        blocks: vec![rate_card, resource_meter, ingest_storage, serving],
    });
}

/// Approximate object-store PUT count for one supertable ingest: one PUT
/// per committed superfile plus one manifest PUT per commit.
pub fn supertable_ingest_puts(n_superfiles: usize) -> u64 {
    n_superfiles as u64 + crate::ingest::supertable::n_commits() as u64
}

/// Flatten cold FTS timings into cost rows.
pub fn cold_from_fts(
    cold: &std::collections::HashMap<&'static str, crate::executors::ColdTiming>,
) -> Vec<ColdQuery> {
    cold.iter()
        .map(|(name, t)| ColdQuery {
            name: (*name).to_string(),
            open_s: t.open.as_secs_f64(),
            search_s: t.search.as_secs_f64(),
        })
        .collect()
}

/// Flatten warm FTS stats into `(name, p50_seconds)` for the cost model.
pub fn warm_from_fts(stats: &[crate::executors::fts::FtsQueryStat]) -> Vec<(String, f64)> {
    stats
        .iter()
        .map(|s| (s.name.to_string(), s.p50.as_secs_f64()))
        .collect()
}

/// Flatten warm SQL query sets into `(name, p50_seconds)`.
pub fn warm_from_sql(sets: &crate::executors::sql::QuerySets) -> Vec<(String, f64)> {
    sets.scalar
        .iter()
        .chain(&sets.tvf)
        .chain(&sets.fts_pushdown)
        .chain(&sets.agg_idx)
        .map(|s| (s.name.to_string(), s.p50.as_secs_f64()))
        .collect()
}

/// Flatten warm vector recall rows into `(label, p50_seconds)`.
pub fn warm_from_vector(rows: &[crate::executors::vector::RecallRow]) -> Vec<(String, f64)> {
    rows.iter()
        .filter_map(|r| {
            r.warm.as_ref().map(|w| {
                let label = if r.params.is_empty() || r.params == "—" {
                    r.target.clone()
                } else {
                    format!("{} ({})", r.target, r.params)
                };
                (label, w.p50_ns / 1e9)
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
    fn parallel_ingest_costs_more_per_second_than_single_writer() {
        let inst = test_instance();
        let single = inst.ingest_compute_usd(10.0, 1);
        let full = inst.ingest_compute_usd(10.0, 8);
        assert!((full / single - 8.0).abs() < 1e-9);
    }

    #[test]
    fn lower_latency_yields_more_queries_per_dollar() {
        let inst = test_instance();
        let fast = inst.per_query_usd(0.001, 1 << 20);
        let slow = inst.per_query_usd(0.010, 1 << 20);
        assert!(slow > fast);
        assert!((slow / fast - 10.0).abs() < 1e-6);
    }

    #[test]
    fn ram_binds_only_when_heap_exceeds_per_core_budget() {
        let inst = test_instance();
        assert!(!inst.ram_binds(1 << 30));
        assert!(inst.ram_binds(3 * (1 << 30)));
    }
}
