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
//!   2. **Object-store requests.** PUTs on ingest, GETs on cold (GET
//!      pricing reserved for a later cold-cost column).
//!   3. **Object-store capacity.** `stored_GB · $/GB-month`.
//!
//! Local NVMe (file-backed disk-cache mmap) is treated as free.

use std::sync::OnceLock;

use crate::report::{Better, Block, Report, Section, metric, text};

/// S3 Standard capacity, USD per GB-month (decimal GB).
const USD_PER_GB_MONTH: f64 = 0.023;
/// USD per PUT request ($5 per 1M).
const USD_PER_PUT: f64 = 5.0e-6;

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
        let cpu_share = f64::from(writers.min(self.vcpu)) / f64::from(self.vcpu.max(1));
        wall_s * self.usd_per_sec() * cpu_share
    }

    fn per_query_usd(&self, p50_s: f64, resident_anon_bytes: u64) -> f64 {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        p50_s * self.usd_per_sec() * cpu_share.max(ram_share)
    }

    fn ram_binds(&self, resident_anon_bytes: u64) -> bool {
        let cpu_share = 1.0 / f64::from(self.vcpu.max(1));
        let ram_share = resident_anon_bytes as f64 / BYTES_PER_GIB / self.ram_gib;
        ram_share > cpu_share
    }
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
}

fn usd(v: f64) -> String {
    if v < 0.01 {
        format!("${:.4}", v)
    } else {
        format!("${:.2}", v)
    }
}

pub fn emit(report: &mut Report, anchor: &str, title: String, c: &CellCost) {
    let inst = Instance::current();

    let compute = inst.ingest_compute_usd(c.ingest_wall_s, c.writers);
    let requests = c.put_count as f64 * USD_PER_PUT;
    let ingest_total = compute + requests;
    let per_million = if c.n_docs > 0 {
        ingest_total / (c.n_docs as f64 / 1.0e6)
    } else {
        0.0
    };
    let stored_gb = c.stored_bytes as f64 / BYTES_PER_GB;
    let storage_month = stored_gb * USD_PER_GB_MONTH;
    let storage_per_million_docs_month = if c.n_docs > 0 {
        storage_month / (c.n_docs as f64 / 1.0e6)
    } else {
        0.0
    };

    let ingest_storage = Block {
        subtitle: format!(
            "Ingest & storage — priced on {} ({} vCPU / {:.0} GiB / {:.0} GB NVMe @ ${:.4}/hr)",
            inst.name, inst.vcpu, inst.ram_gib, inst.nvme_gb, inst.usd_per_hour,
        ),
        headers: vec![
            "Component".into(),
            "Cost".into(),
            "Per-unit".into(),
        ],
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
                text(format!("{}/1M docs·mo", usd(storage_per_million_docs_month))),
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
        note: "First-cut cost model. Compute is priced at the instance's marginal rate on the \
               binding resource (ingest = CPU-time of the build; serving = one core per query, or \
               resident anonymous heap if tighter). Object-store requests are $5/1M PUT; capacity is \
               $0.023/GB-month. Local NVMe — and therefore the file-backed disk cache — is free. \
               Δ is vs the previous run."
            .into(),
        blocks: vec![ingest_storage, serving],
    });
}

/// Approximate object-store PUT count for one supertable ingest: one PUT
/// per committed superfile plus one manifest PUT per commit.
pub fn supertable_ingest_puts(n_superfiles: usize) -> u64 {
    n_superfiles as u64 + crate::ingest::supertable::n_commits() as u64
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
