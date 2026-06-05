//! `supertable_vec_search` — vector kNN on the shared combined supertable.

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use criterion::Criterion;
use infino::supertable::Supertable;
use infino::supertable::query::vector::VectorSearchOptions;

use crate::bench::supertable::query::vector_topk_global;
use crate::corpus::{Calibrated, grading};
use crate::fixture::supertable as fixture;
use crate::ingest::supertable;
use crate::tiers::{self, Tier};
use crate::{markdown, rss};

const TOP_K: usize = 10;
const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];

const SUPERTABLE_PROBES_PER_SEG: &[usize] = &[1, 2, 4, 8, 12, 16, 32, 64, 128];
const CALIBRATION_P50_ITERS: usize = 7;

const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
const CORRECTNESS_NPROBE: usize = 16;

static CALIBRATIONS: OnceLock<Calibrations> = OnceLock::new();

pub mod group_name {
    pub const SUPERTABLE_VEC_SEARCH: &str = "supertable_vec_hot_search";
}

struct Calibrations {
    supertable: [Option<Calibrated>; 3],
}

fn mean_recall(
    st: &Supertable,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    options: VectorSearchOptions,
) -> f32 {
    let mut sum = 0f32;
    for (q, t) in queries.iter().zip(truths) {
        let hits = vector_topk_global(st, q, TOP_K, options);
        let truth_set: std::collections::HashSet<u32> = t.iter().copied().collect();
        let recall = if t.is_empty() {
            1.0
        } else {
            hits.iter().filter(|id| truth_set.contains(id)).count() as f32 / t.len() as f32
        };
        sum += recall;
    }
    sum / queries.len() as f32
}

fn assert_correctness(st: &Supertable, g: &grading::SupertableGrading) -> f32 {
    let opts = VectorSearchOptions::new().with_nprobe(CORRECTNESS_NPROBE);
    let mean_recall = mean_recall(st, &g.correctness_queries, &g.correctness_gt, opts);
    assert!(
        mean_recall >= CORRECTNESS_RECALL_FLOOR,
        "supertable mean recall@{TOP_K} at correctness config \
         (p={CORRECTNESS_NPROBE}) \
         below floor: {mean_recall:.3} < {CORRECTNESS_RECALL_FLOOR:.3}"
    );
    mean_recall
}

fn measure_p50_micros(st: &Supertable, query: &[f32], options: VectorSearchOptions) -> f32 {
    let mut samples = Vec::with_capacity(CALIBRATION_P50_ITERS);
    for _ in 0..CALIBRATION_P50_ITERS {
        let t0 = Instant::now();
        let _ = vector_topk_global(st, query, TOP_K, options);
        samples.push(t0.elapsed().as_secs_f32() * 1e6);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn calibration_grid(st: &Supertable, queries: &[Vec<f32>], truths: &[Vec<u32>]) -> Vec<Calibrated> {
    let min_target = RECALL_TARGETS.iter().copied().fold(f32::INFINITY, f32::min);
    let refine = VectorSearchOptions::RERANK_MULT;
    let total = SUPERTABLE_PROBES_PER_SEG.len();
    let mut seen = 0usize;
    let mut points = Vec::with_capacity(total);
    for &probe in SUPERTABLE_PROBES_PER_SEG {
        seen += 1;
        let opts = VectorSearchOptions::new().with_nprobe(probe);
        let t0 = Instant::now();
        let recall = mean_recall(st, queries, truths, opts);
        let recall_secs = t0.elapsed().as_secs_f32();
        let p50_micros = if recall >= min_target {
            measure_p50_micros(st, &queries[0], opts)
        } else {
            f32::INFINITY
        };
        let p50_msg = if p50_micros.is_finite() {
            format!("{p50_micros:.0}us")
        } else {
            "not measured".to_string()
        };
        eprintln!(
            "  [{seen:02}/{total}] p={probe:<3} recall={recall:.3} recall_eval={recall_secs:.1}s p50={p50_msg}"
        );
        points.push(Calibrated {
            probe,
            refine,
            recall,
            p50_micros,
        });
        if targets_satisfied(&points) {
            eprintln!(
                "  calibration early-stop: all recall targets satisfied after p={probe}; \
                 skipping higher nprobe rows"
            );
            break;
        }
    }
    points
}

fn targets_satisfied(points: &[Calibrated]) -> bool {
    RECALL_TARGETS
        .iter()
        .all(|&target| points.iter().any(|p| p.recall >= target))
}

fn select_at_target(points: &[Calibrated], target_recall: f32) -> Option<Calibrated> {
    let mut best: Option<Calibrated> = None;
    let mut peak_recall = 0f32;
    for &cand in points {
        peak_recall = peak_recall.max(cand.recall);
        if cand.recall < target_recall {
            continue;
        }
        best = match best {
            None => Some(cand),
            Some(b) if cand.p50_micros < b.p50_micros => Some(cand),
            Some(b) => Some(b),
        };
    }
    if best.is_none() {
        eprintln!(
            "    [supertable] no point hit recall ≥ {target_recall:.2}; peak observed = {peak_recall:.3}"
        );
    }
    best
}

fn calibrations(st: &Supertable, g: &grading::SupertableGrading) -> &'static Calibrations {
    CALIBRATIONS.get_or_init(|| {
        eprintln!(
            "[supertable_vec_search] calibrating vector search at recall targets {RECALL_TARGETS:?}..."
        );
        let points = calibration_grid(st, &g.calibration_queries, &g.calibration_gt);
        let mut s: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            s[i] = select_at_target(&points, target);
            eprintln!("  recall ≥ {target:.2} | vector search: {:?}", s[i]);
        }
        Calibrations { supertable: s }
    })
}

pub fn bench(c: &mut Criterion) {
    if !fixture::criterion_filter_selects(
        &["supertable_vec", "supertable_vec_search"],
        &[
            "supertable_vec_hot_search",
            "supertable_vec_cold_search_real_s3",
        ],
    ) {
        return;
    }
    fixture::ensure_ingest_for_search("vector correctness/search");
    let g = grading::supertable_grading();
    let st = fixture::search_table();
    eprintln!(
        "[supertable_vec] correctness: {} docs, {} superfiles",
        supertable::N_DOCS,
        fixture::ensure_ingest_for_search("reporting").n_superfiles
    );
    let recall = assert_correctness(st, g);
    eprintln!(
        "[supertable_vec] correctness OK: vector recall@{TOP_K} = {recall:.3} (≥ {CORRECTNESS_RECALL_FLOOR:.2})"
    );

    // Warm the shared in-process cache the way a real serving node
    // does: serve a few queries through the public API to kick off the
    // background full-segment fills, then block on the public
    // `Supertable::wait_until_warm` until every segment is actually
    // mmap-promoted. The query replay alone does NOT make the tier hot —
    // it leaves the segments backed by the lazy object-store source, so
    // the "hot" measurement would really time live S3 range GETs
    // (~seconds). `wait_until_warm` is the documented warm-readiness
    // primitive (see supertable.md); it both waits for and drives
    // promotion to completion, which is exactly what a node does before
    // taking traffic.
    eprintln!("[supertable_vec] warming cache by serving queries (public API)...");
    let warm_t0 = Instant::now();
    let warm_opts = VectorSearchOptions::new().with_nprobe(CORRECTNESS_NPROBE);
    for query in g.correctness_queries.iter() {
        let _ = vector_topk_global(st, query, TOP_K, warm_opts);
    }
    eprintln!(
        "[supertable_vec] served {} warm-up queries in {:.1}s; waiting for mmap promotion...",
        g.correctness_queries.len(),
        warm_t0.elapsed().as_secs_f32()
    );
    st.wait_until_warm(Duration::from_secs(180))
        .expect("supertable cache failed to reach warm (mmap-promoted) state");
    eprintln!(
        "[supertable_vec] cache fully warm (all segments mmap-promoted) in {:.1}s",
        warm_t0.elapsed().as_secs_f32()
    );

    let cal = calibrations(st, g);
    let qs = &g.calibration_queries;

    let mut g_hot = c.benchmark_group(tiers::search_group_name("supertable_vec", Tier::Hot, None));
    g_hot.sample_size(10);
    let rss_sample = rss::PeakSampler::start_default();

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        if let Some(c_st) = cal.supertable[i] {
            let p = c_st.probe;
            g_hot.bench_function(format!("supertable_{label}"), |b| {
                let q = &qs[0];
                let opts = VectorSearchOptions::new().with_nprobe(p);
                b.iter(|| {
                    let hits = vector_topk_global(st, black_box(q), TOP_K, opts);
                    black_box(hits)
                });
            });
        }
    }

    g_hot.finish();
    let stats = rss_sample.stop_stats();
    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        if cal.supertable[i].is_some() {
            let bid = format!("supertable_{label}");
            let _ = rss::write_rss_stats(group_name::SUPERTABLE_VEC_SEARCH, &bid, stats);
        }
    }

    bench_object_store_tiers(c, cal, qs);
    emit_markdown(cal);
}

fn bench_object_store_tiers(c: &mut Criterion, cal: &Calibrations, qs: &[Vec<f32>]) {
    let q = &qs[0];
    let storage_label = fixture::storage_label();
    let idx_bytes = Some(fixture::total_index_bytes());

    // Cold tier only against object storage. The mmap-promoted "warm"
    // tier was dropped: nothing is pinned in memory, so its latency is
    // indistinguishable from the in-process hot tier measured above, and
    // it was the sole reason the bench reached for an internal warm hook.
    let tier = Tier::Cold;
    let mut g = c.benchmark_group(tiers::search_group_name(
        "supertable_vec",
        tier,
        Some(storage_label),
    ));
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(30));

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let Some(c_st) = cal.supertable[i] else {
            continue;
        };
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let p = c_st.probe;
        let opts = VectorSearchOptions::new().with_nprobe(p);
        let bench_id = format!("supertable_{label}");

        let storage = fixture::storage();
        let query = q.clone();
        g.bench_function(&bench_id, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let (cache_dir, cache) =
                        tiers::fresh_supertable_search_cache(Arc::clone(&storage), idx_bytes);
                    let consumer_opts = tiers::consumer_options(
                        supertable::combined_options(None),
                        Arc::clone(&storage),
                        cache.clone(),
                    );
                    // COLD TIER = "first query after the reader is up, before
                    // any segment data is cached." It must NOT include the
                    // manifest open.
                    //
                    // Why: `Supertable::open()` reads the manifest from object
                    // storage exactly once and pins it in memory behind an
                    // `ArcSwap<Manifest>` for the life of the reader. Every
                    // subsequent query reuses that in-memory manifest via an
                    // Arc clone — zero object-store access. So in production the
                    // manifest read is a one-time, process-lifetime cost, never
                    // a per-query cost.
                    //
                    // Each iteration still gets a FRESH disk cache (above), which
                    // is what legitimately makes the segment *data* cold — the
                    // query below must range-GET codes/doc_ids/vectors from S3.
                    // We open the consumer (one-time manifest read) OUTSIDE the
                    // timed region and start the clock only around the query, so
                    // the number reflects a real cold per-query latency rather
                    // than a benchmark artifact of re-opening the table every
                    // iteration. open + query are both public API; we only move
                    // where the timer starts.
                    let st = tiers::open_consumer(consumer_opts);
                    let t0 = Instant::now();
                    let _ = vector_topk_global(&st, &query, TOP_K, opts);
                    let elapsed = t0.elapsed();
                    total += elapsed;
                    drop(st);
                    drop(cache);
                    drop(cache_dir);
                }
                total
            });
        });
    }
    g.finish();
}

fn emit_markdown(cal: &Calibrations) {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_VEC_SEARCH;
    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable vector — search ({} docs × dim={}, calibrated at recall targets)\n\n",
        supertable::N_DOCS,
        crate::corpus::DIM
    ));
    body.push_str(
        "hot = warm steady state: every segment mmap-promoted via the public `Supertable::wait_until_warm` before timing, so reads hit resident pages (no object-store GETs). cold = fresh disk cache → object-store range GETs (s3s-fs or `INFINO_REAL_S3_BUCKET`), excluding the one-time manifest open.\n\n",
    );
    body.push_str(
        "| Recall target | (p/seg, r) | hot | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|---------------|------------|-----|------|----------|------------|---------|------------|\n",
    );

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let row_target = format!("{target:.2}");
        let bid = format!("supertable_{label}");
        let (cell, hot, cold, rss_cell, median_rss, p90_rss, rss_delta) = match cal.supertable[i] {
            Some(c) => {
                let peak = rss::read_peak_rss_bytes(group, &bid);
                (
                    format!("(p={}, r={})", c.probe, c.refine),
                    read_mean_ns(group, &bid),
                    markdown::read_tier_mean_ns("supertable_vec", "cold", &bid),
                    peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into()),
                    rss::fmt_median_rss(group, &bid),
                    rss::fmt_p90_rss(group, &bid),
                    rss::fmt_peak_rss_delta(group, &bid),
                )
            }
            None => (
                "—".into(),
                None,
                None,
                "—".into(),
                "—".into(),
                "—".into(),
                "—".into(),
            ),
        };
        body.push_str(&format!(
            "| {row_target} | {cell} | {} | {} | {rss_cell} | {median_rss} | {p90_rss} | {rss_delta} |\n",
            hot.map(fmt_time).unwrap_or_else(|| "—".into()),
            cold.map(fmt_time).unwrap_or_else(|| "—".into()),
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/supertable/search".into(),
        body,
    });
}
