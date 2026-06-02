//! `supertable_all_build` — time one object-storage ingest.

use std::time::Duration;

use criterion::{Criterion, Throughput};

use crate::fixture::supertable as fixture;
use crate::ingest::supertable;
use crate::{markdown, rss};

const BUILD_MEASUREMENT_TIME: Duration = Duration::from_secs(60 * 60);

pub mod group_name {
    pub const SUPERTABLE_ALL_BUILD: &str = "supertable_all_build";
}

pub fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group(group_name::SUPERTABLE_ALL_BUILD);
    g.sample_size(10);
    g.measurement_time(BUILD_MEASUREMENT_TIME);
    g.throughput(Throughput::Elements(supertable::N_DOCS as u64));

    let rss_sample = rss::PeakSampler::start_default();
    let bench_id = format!("supertable_{}docs", supertable::N_DOCS);
    g.bench_function(bench_id.clone(), |b| {
        b.iter_custom(|iters| {
            fixture::ensure_ingest("ingest timing");
            let ns = fixture::ingest_build_nanos();
            Duration::from_nanos(ns as u64) * (iters as u32)
        });
    });

    g.finish();
    let stats = rss_sample.stop_stats();
    let _ = rss::write_rss_stats(group_name::SUPERTABLE_ALL_BUILD, &bench_id, stats);

    emit_markdown(&bench_id);
}

fn emit_markdown(bench: &str) {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_ALL_BUILD;
    let ns = read_mean_ns(group, bench);
    let peak_rss = rss::read_peak_rss_bytes(group, bench);
    let n_superfiles = fixture::ensure_ingest("markdown").n_superfiles;

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable combined FTS + vector — ingest ({} docs × dim={}, {} commits → {} superfiles)\n\n",
        supertable::N_DOCS,
        crate::corpus::DIM,
        supertable::N_COMMIT_CHUNKS,
        n_superfiles
    ));
    body.push_str(
        "| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|--------|------|------------|----------|------------|---------|------------|\n",
    );
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((supertable::N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let median_rss = rss::fmt_median_rss(group, bench);
    let p90_rss = rss::fmt_p90_rss(group, bench);
    let rss_delta = rss::fmt_peak_rss_delta(group, bench);
    body.push_str(&format!(
        "| supertable | {time} | {thrpt} | {rss_cell} | {median_rss} | {p90_rss} | {rss_delta} |\n"
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/supertable/ingest".into(),
        body,
    });
}
