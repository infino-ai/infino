//! `supertable_fts_search` — FTS on the shared combined supertable.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::Criterion;
use infino::superfile::fts::reader::BoolMode;

use crate::fixture::supertable as fixture;
use crate::ingest::supertable;
use crate::tiers::{self, Tier};
use crate::{markdown, rss};

const TOP_K: usize = 10;

pub const FTS_SEARCH_IDS: &[&str] = &[
    "single_rare_supertable_top10",
    "single_common_supertable_top10",
    "two_term_or_supertable_top10",
    "three_wide_or_supertable_top10",
    "three_similar_or_supertable_top10",
    "five_term_or_supertable_top10",
    "ten_term_or_supertable_top10",
    "prefix_supertable_top10",
];

pub const FTS_QUERIES: &[(&str, &str)] = &[
    ("single_rare", "term09999"),
    ("single_common", "term00001"),
    ("two_term_or", "term00001 term00050"),
    ("three_wide_or", "term00001 term00050 term00100"),
    ("three_similar_or", "term00050 term00051 term00052"),
    (
        "five_term_or",
        "term00050 term00051 term00052 term00053 term00054",
    ),
    (
        "ten_term_or",
        "term00050 term00051 term00052 term00053 term00054 \
         term00055 term00056 term00057 term00058 term00059",
    ),
];

pub mod group_name {
    pub const SUPERTABLE_FTS_SEARCH: &str = "supertable_fts_hot_search";
}

fn assert_fts_self_consistent(st: &infino::supertable::Supertable) {
    let probe_doc_id = (supertable::N_DOCS / 2) as u32;
    let probe_token = format!("doc{probe_doc_id:07}");
    let hits = st
        .bm25_search(supertable::TEXT_COLUMN, &probe_token, TOP_K, BoolMode::Or)
        .expect("bm25");
    assert_eq!(
        hits.len(),
        1,
        "df=1 token {probe_token:?} should return exactly one hit; got {}",
        hits.len()
    );
    assert!(
        hits[0].score > 0.0,
        "df=1 score must be positive; got {}",
        hits[0].score
    );

    let hits = st
        .bm25_search(supertable::TEXT_COLUMN, "term00001", TOP_K, BoolMode::Or)
        .expect("bm25");
    assert_eq!(hits.len(), TOP_K, "common term should fill top-{TOP_K}");
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results must be sorted by score desc; got {} then {}",
            w[0].score,
            w[1].score
        );
    }
}

pub fn bench(c: &mut Criterion) {
    if !fixture::criterion_filter_selects(
        &["supertable_fts", "supertable_fts_search"],
        &[
            "supertable_fts_hot_search",
            "supertable_fts_cold_search_real_s3",
        ],
    ) {
        return;
    }
    fixture::ensure_ingest_for_search("FTS correctness/search");
    let st = fixture::search_table();
    eprintln!("[supertable_fts_search] correctness on object-store supertable...");
    assert_fts_self_consistent(st);
    eprintln!("[supertable_fts_search] correctness OK");

    // Drive the cache to the warm (mmap-promoted) steady state through
    // the public API before timing the hot tier. Without this the first
    // touch of each segment faults through the lazy object-store source,
    // so "hot" would fold in live S3 range GETs instead of resident
    // mmap reads. See `Supertable::wait_until_warm` / supertable.md.
    eprintln!("[supertable_fts_search] warming cache (mmap promotion)...");
    let warm_t0 = std::time::Instant::now();
    for (_, q) in FTS_QUERIES {
        let _ = st.bm25_search(supertable::TEXT_COLUMN, q, TOP_K, BoolMode::Or);
    }
    st.wait_until_warm(Duration::from_secs(180))
        .expect("supertable cache failed to reach warm (mmap-promoted) state");
    eprintln!(
        "[supertable_fts_search] cache fully warm in {:.1}s",
        warm_t0.elapsed().as_secs_f32()
    );

    let mut g = c.benchmark_group(tiers::search_group_name("supertable_fts", Tier::Hot, None));
    g.sample_size(10);
    let rss_sample = rss::PeakSampler::start_default();

    for (name, q) in FTS_QUERIES {
        let name = *name;
        let q = *q;
        g.bench_function(format!("{name}_supertable_top10"), |b| {
            b.iter(|| {
                let hits =
                    st.bm25_search(supertable::TEXT_COLUMN, black_box(q), TOP_K, BoolMode::Or);
                black_box(hits)
            });
        });
    }

    g.bench_function("prefix_supertable_top10", |b| {
        b.iter(|| {
            let hits = st.bm25_search_prefix(supertable::TEXT_COLUMN, black_box("term0009"), TOP_K);
            black_box(hits)
        });
    });

    g.finish();
    let stats = rss_sample.stop_stats();
    for bid in FTS_SEARCH_IDS {
        let _ = rss::write_rss_stats(group_name::SUPERTABLE_FTS_SEARCH, bid, stats);
    }

    bench_object_store_tiers(c);
    emit_markdown();
}

fn bench_object_store_tiers(c: &mut Criterion) {
    let storage_label = fixture::storage_label();
    let idx_bytes = Some(fixture::total_index_bytes());

    // Only the cold tier is measured against object storage. The warm
    // tier (mmap-promoted) was dropped: with nothing pinned in memory,
    // its latency is indistinguishable from the in-process hot tier
    // measured above, so it carried no signal — and it was the only
    // reason the bench reached for an internal warm hook. A real client
    // only ever sees "cold first touch" vs "already-served hot".
    let tier = Tier::Cold;
    let mut g = c.benchmark_group(tiers::search_group_name(
        "supertable_fts",
        tier,
        Some(storage_label),
    ));
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(30));

    for (name, q) in FTS_QUERIES {
        let bench_id = format!("{name}_supertable_top10");
        let storage = fixture::storage();
        let query = *q;
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
                    let st = tiers::open_consumer(consumer_opts);
                    let t0 = std::time::Instant::now();
                    let _ = st
                        .bm25_search(supertable::TEXT_COLUMN, query, TOP_K, BoolMode::Or)
                        .expect("cold bm25");
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

    g.bench_function("prefix_supertable_top10", |b| {
        let storage = fixture::storage();
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
                let st = tiers::open_consumer(consumer_opts);
                let t0 = std::time::Instant::now();
                let _ = st
                    .bm25_search_prefix(supertable::TEXT_COLUMN, "term0009", TOP_K)
                    .expect("cold prefix");
                let elapsed = t0.elapsed();
                total += elapsed;
                drop(st);
                drop(cache);
                drop(cache_dir);
            }
            total
        });
    });

    g.finish();
}

fn emit_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_FTS_SEARCH;
    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable FTS — search ({} docs, shared combined supertable)\n\n",
        supertable::N_DOCS
    ));
    body.push_str(
        "hot = warm steady state: every segment mmap-promoted via the public `Supertable::wait_until_warm` before timing, so reads hit resident pages (no object-store GETs). cold = fresh disk cache → object-store range GETs (s3s-fs or `INFINO_REAL_S3_BUCKET`), excluding the one-time manifest open.\n\n",
    );
    body.push_str(
        "| Query          | hot        | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|----------------|------------|------------|-----------|------------|-----------|------------|\n",
    );

    for q in [
        "single_rare",
        "single_common",
        "two_term_or",
        "three_wide_or",
        "three_similar_or",
        "five_term_or",
        "ten_term_or",
        "prefix",
    ] {
        let bid = format!("{q}_supertable_top10");
        let hot = read_mean_ns(group, &bid);
        let cold = markdown::read_tier_mean_ns("supertable_fts", "cold", &bid);
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, &bid);
        let p90_rss = rss::fmt_p90_rss(group, &bid);
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {} | {} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n",
            hot.map(fmt_time).unwrap_or_else(|| "—".into()),
            cold.map(fmt_time).unwrap_or_else(|| "—".into()),
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/supertable/search".into(),
        body,
    });
}
