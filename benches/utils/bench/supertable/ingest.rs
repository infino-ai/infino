//! Supertable ingest timing — three apples-to-apples shapes:
//! `supertable_fts_build` (vs Tantivy), `supertable_vec_build` (vs Lance
//! vector-only), and `supertable_all_build` (combined, vs combined Lance).

use std::time::Duration;

use criterion::{Criterion, Throughput};

use crate::fixture::supertable as fixture;
use crate::ingest::supertable;
use crate::{markdown, rss};

const BUILD_MEASUREMENT_TIME: Duration = Duration::from_secs(60 * 60);

pub mod group_name {
    pub const SUPERTABLE_ALL_BUILD: &str = "supertable_all_build";
    pub const SUPERTABLE_FTS_BUILD: &str = "supertable_fts_build";
    pub const SUPERTABLE_VEC_BUILD: &str = "supertable_vec_build";
}

pub fn bench(c: &mut Criterion) {
    // FTS-only and vector-only first so each RSS window opens before the
    // combined build (and the shared search consumer) makes anything resident.
    run_build(
        c,
        group_name::SUPERTABLE_FTS_BUILD,
        &format!("supertable_fts_{}docs", supertable::N_DOCS),
        "FTS-only",
        fixture::fts_ingest_build_nanos,
        || {
            fixture::ensure_fts_ingest("ingest timing");
        },
        fixture::fts_ingest_recorded,
        || fixture::ensure_fts_ingest("markdown").n_superfiles,
    );
    run_build(
        c,
        group_name::SUPERTABLE_VEC_BUILD,
        &format!("supertable_vec_{}docs", supertable::N_DOCS),
        "vector-only",
        fixture::vector_ingest_build_nanos,
        || {
            fixture::ensure_vector_ingest("ingest timing");
        },
        fixture::vector_ingest_recorded,
        || fixture::ensure_vector_ingest("markdown").n_superfiles,
    );
    run_build(
        c,
        group_name::SUPERTABLE_ALL_BUILD,
        &format!("supertable_{}docs", supertable::N_DOCS),
        "combined FTS + vector",
        fixture::ingest_build_nanos,
        || {
            fixture::ensure_ingest("ingest timing");
        },
        fixture::ingest_recorded,
        || fixture::ensure_ingest("markdown").n_superfiles,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_build(
    c: &mut Criterion,
    group: &str,
    bench_id: &str,
    shape: &str,
    build_nanos: fn() -> f64,
    ensure: impl Fn(),
    was_built: impl Fn() -> bool,
    n_superfiles: impl Fn() -> usize,
) {
    let mut g = c.benchmark_group(group);
    g.sample_size(10);
    g.measurement_time(BUILD_MEASUREMENT_TIME);
    g.throughput(Throughput::Elements(supertable::N_DOCS as u64));

    let rss_sample = rss::PeakSampler::start_default();
    g.bench_function(bench_id, |b| {
        b.iter_custom(|iters| {
            ensure();
            let ns = build_nanos();
            Duration::from_nanos(ns as u64) * (iters as u32)
        });
    });
    g.finish();
    if !was_built() {
        return;
    }
    let stats = rss_sample.stop_stats();
    let _ = rss::write_rss_stats(group, bench_id, stats);

    emit_markdown(group, bench_id, shape, &n_superfiles());
}

fn emit_markdown(group: &str, bench: &str, shape: &str, n_superfiles: &usize) {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let ns = read_mean_ns(group, bench);
    let peak_rss = rss::read_peak_rss_bytes(group, bench);

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable {} — ingest ({} docs × dim={}, {} commits → {} superfiles)\n\n",
        shape,
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
        anchor_id: format!("bench/supertable/ingest/{group}"),
        body,
    });
}
