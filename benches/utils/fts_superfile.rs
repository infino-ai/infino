//! Infino-only FTS bench for the superfile layer:
//!
//!   ingest timing (single-thread + rayon-sharded multi-thread)
//! + 7-query search timing
//! + 3 per-algorithm (WAND+BMW vs MaxScore+BMM) probes
//! + correctness gates (BMW-vs-brute-force, df=1 + common-term ordering)
//!
//! Every phase uses the production path: [`SuperfileBuilder`] → unified
//! `.parquet` → [`SuperfileReader`] → `bm25_search` / embedded [`FtsReader`].
//! Hot opens the finished `.parquet` in memory; warm/cold commit the same bytes
//! to object storage and read through [`DiskCacheStore::reader`].
//!
//! Pinned to 1M-doc Zipfian (200 tokens/doc, 10K vocab). The
//! single-superfile shape is rarely much larger in production —
//! `benches/fts/supertable.rs` covers the 10M+ supertable scale.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts                            # all FTS benches
//! cargo bench --bench fts -- superfile_fts_build     # only superfile ingest
//! cargo bench --bench fts -- superfile_fts_search    # only superfile search
//! cargo bench --bench fts -- _build                  # ingest across superfile + supertable
//! ```
//!
//! Correctness phase runs unconditionally on every invocation
//! (criterion filters skip timing, not setup), so a filter to
//! `superfile_fts_search` still validates the BMW oracle before
//! timing kicks in.

use crate::tiers::{self, Tier};
use crate::{corpus, markdown, rss};
use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group};
use infino::superfile::SuperfileReader;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use infino::superfile::fts::reader::{BoolMode, OrAlgo};
use infino::test_helpers::default_tokenizer;
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

// ─── Constants ────────────────────────────────────────────────────────

/// Doc count for every FTS-superfile bench. Superfile shape → 1M.
const N_DOCS: usize = corpus::SUPERFILE_DOCS;

const ID_COLUMN: &str = "doc_id";
const FTS_COLUMN: &str = "title";

// ─── Fixtures ────────────────────────────────────────────────────────

static DOCS: OnceLock<Vec<String>> = OnceLock::new();
static SUPERFILE_BYTES: OnceLock<Vec<u8>> = OnceLock::new();
static SUPERFILE_OBJECT: OnceLock<tiers::SuperfileCommitted> = OnceLock::new();

fn superfile_bytes() -> &'static [u8] {
    SUPERFILE_BYTES.get_or_init(|| build_superfile_bytes(docs()))
}

fn superfile_object() -> &'static tiers::SuperfileCommitted {
    SUPERFILE_OBJECT.get_or_init(|| {
        let blob = Bytes::from(superfile_bytes().to_vec());
        eprintln!(
            "[superfile_fts] committing {N_DOCS} docs to object storage for warm/cold tiers \
             (production .parquet, {} MiB)...",
            blob.len() / (1024 * 1024)
        );
        tiers::block_on(tiers::commit_superfile(&blob))
    })
}

const TIER_OR_QUERIES: &[(&str, &[&str])] = &[
    ("single_common", &["term00001"]),
    ("two_term_or", &["term00001", "term00050"]),
    ("three_wide_or", &["term00001", "term00050", "term00100"]),
];

fn docs() -> &'static [String] {
    DOCS.get_or_init(|| corpus::generate_text_corpus(N_DOCS, 1))
        .as_slice()
}

fn superfile_reader() -> SuperfileReader {
    SuperfileReader::open(Bytes::from(superfile_bytes().to_vec())).expect("open superfile")
}

// ─── Builder (production SuperfileBuilder) ───────────────────────────

fn supertable_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(ID_COLUMN, DataType::Decimal128(38, 0), false),
        Field::new(FTS_COLUMN, DataType::LargeUtf8, false),
    ]))
}

/// Unified `.parquet` (Parquet body + embedded FTS blob + `inf.*` KV) — same
/// path as supertable commit and `vector_superfile`.
fn build_superfile_bytes(docs: &[String]) -> Vec<u8> {
    let schema = supertable_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![FtsConfig {
            column: FTS_COLUMN.into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
    const CHUNK: usize = 65_536;
    let mut start = 0;
    while start < docs.len() {
        let len = CHUNK.min(docs.len() - start);
        let ids: Decimal128Array = (start as u64..(start + len) as u64)
            .map(|i| Some(i as i128))
            .collect::<Decimal128Array>()
            .with_precision_and_scale(38, 0)
            .expect("decimal128");
        let titles = LargeStringArray::from(
            docs[start..start + len]
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        );
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(titles)])
                .expect("RecordBatch");
        builder.add_batch(&batch, &[]).expect("add_batch");
        start += len;
    }
    builder.finish().expect("SuperfileBuilder::finish")
}

/// Rayon-sharded parallel build: each shard runs its own
/// [`SuperfileBuilder`] and emits a self-contained `.parquet` — the same
/// multi-segment shape supertable commit produces.
fn build_superfiles_rayon(docs: &[String]) -> Vec<Vec<u8>> {
    let n_shards = rayon::current_num_threads();
    let docs_per_shard = docs.len().div_ceil(n_shards);
    docs.chunks(docs_per_shard)
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(shard_idx, chunk)| {
            let id_base = shard_idx * docs_per_shard;
            build_superfile_bytes_with_id_base(chunk, id_base)
        })
        .collect()
}

fn build_superfile_bytes_with_id_base(docs: &[String], id_base: usize) -> Vec<u8> {
    let schema = supertable_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![FtsConfig {
            column: FTS_COLUMN.into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
    let ids: Decimal128Array = (id_base as u64..(id_base + docs.len()) as u64)
        .map(|i| Some(i as i128))
        .collect::<Decimal128Array>()
        .with_precision_and_scale(38, 0)
        .expect("decimal128");
    let titles = LargeStringArray::from(
        docs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("RecordBatch");
    builder.add_batch(&batch, &[]).expect("add_batch");
    builder.finish().expect("finish")
}

// ─── Correctness ──────────────────────────────────────────────────────

fn assert_superfile_self_consistent(reader: &SuperfileReader) {
    let hits = corpus::block_on_inmem(
        reader.bm25_search(FTS_COLUMN, "doc0500000", 10, BoolMode::Or),
    )
    .expect("search df=1");
    assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
    assert_eq!(hits[0].0, 500_000, "doc0500000 should match doc_id 500000");

    let hits = corpus::block_on_inmem(
        reader.bm25_search(FTS_COLUMN, "term00001", 10, BoolMode::Or),
    )
    .expect("search common");
    assert_eq!(hits.len(), 10, "common term should fill top-10");
    for w in hits.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "results must be sorted by score desc; got {} then {}",
            w[0].1,
            w[1].1
        );
    }
}

fn assert_bmw_matches_brute_force(reader: &SuperfileReader) -> usize {
    let battery: &[(&str, &[&str])] = &[
        ("single_rare", &["term09999"]),
        ("single_common", &["term00001"]),
        ("two_term_or", &["term00001", "term00050"]),
        ("three_wide_or", &["term00001", "term00050", "term00100"]),
        ("three_similar_or", &["term00050", "term00051", "term00052"]),
        (
            "five_term_or",
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
        ),
    ];
    const SCORE_EPSILON: f32 = 1e-4;

    for (label, terms) in battery {
        let bmw_top10: Vec<(u32, f32)> = corpus::block_on_inmem(
            reader.bm25_search_pretokenized(FTS_COLUMN, terms, 10, BoolMode::Or),
        )
        .expect("bmw search");
        let mut brute_full = corpus::block_on_inmem(
            reader.bm25_search_pretokenized(FTS_COLUMN, terms, usize::MAX, BoolMode::Or),
        )
        .expect("brute-force search");
        brute_full.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let brute_top10: Vec<(u32, f32)> = brute_full.into_iter().take(10).collect();

        assert_eq!(
            bmw_top10.len(),
            brute_top10.len(),
            "result lengths must match on {label}: BMW {} vs brute {}",
            bmw_top10.len(),
            brute_top10.len()
        );
        for i in 0..bmw_top10.len() {
            let (bmw_doc, bmw_score) = bmw_top10[i];
            let (brute_doc, brute_score) = brute_top10[i];
            let diff = (bmw_score - brute_score).abs();
            if diff > SCORE_EPSILON {
                let bmw_seq: Vec<f32> = bmw_top10.iter().map(|(_, s)| *s).collect();
                let brute_seq: Vec<f32> = brute_top10.iter().map(|(_, s)| *s).collect();
                panic!(
                    "BMW vs brute-force score divergence at position {i} on {label} ({terms:?}):\n  \
                     BMW score = {bmw_score} (doc {bmw_doc})\n  \
                     brute score = {brute_score} (doc {brute_doc})\n  \
                     diff = {diff} > epsilon {SCORE_EPSILON}\n  \
                     BMW scores  : {bmw_seq:?}\n  \
                     brute scores: {brute_seq:?}"
                );
            }
        }
    }
    battery.len()
}

// ─── Bench helpers ────────────────────────────────────────────────────

fn bench_infino(
    c: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    r: &SuperfileReader,
    terms: &'static [&'static str],
    mode: BoolMode,
) {
    c.bench_function(format!("{name}_infino_top10"), |b| {
        b.iter(|| {
            let hits = corpus::block_on_inmem(
                r.bm25_search_pretokenized(black_box(FTS_COLUMN), black_box(terms), black_box(10), mode),
            )
            .expect("bm25 search");
            black_box(hits)
        });
    });
}

fn bench_per_algo_probe(
    c: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    r: &SuperfileReader,
    terms: &'static [&'static str],
) {
    let fts = r.fts().expect("FTS subsection");
    c.bench_function(format!("{name}_wand_top10"), |b| {
        b.iter(|| {
            let hits = corpus::block_on_inmem(fts.search_with_algo_for_bench(
                black_box(FTS_COLUMN),
                black_box(terms),
                black_box(10),
                OrAlgo::WandBmw,
            ))
            .expect("WAND+BMW search");
            black_box(hits)
        });
    });
    c.bench_function(format!("{name}_bmm_top10"), |b| {
        b.iter(|| {
            let hits = corpus::block_on_inmem(fts.search_with_algo_for_bench(
                black_box(FTS_COLUMN),
                black_box(terms),
                black_box(10),
                OrAlgo::Bmm,
            ))
            .expect("MaxScore+BMM search");
            black_box(hits)
        });
    });
}

// ─── Bench entry ──────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    eprintln!("[fts/superfile] correctness: building superfile ({N_DOCS} docs)...");
    let r = superfile_reader();
    assert_superfile_self_consistent(&r);
    let n_bmw = assert_bmw_matches_brute_force(&r);
    eprintln!(
        "[fts/superfile] correctness OK: superfile self-consistent + {n_bmw} queries BMW==brute-force"
    );

    {
        let n = N_DOCS;
        let docs_for_ingest = docs();
        let mut g = c.benchmark_group("superfile_fts_build");
        g.sample_size(10);
        g.throughput(Throughput::Elements(n as u64));

        let one_thread_id = format!("infino_1thread_{n}docs");
        let rayon_id = format!("infino_rayon_default_threads_{n}docs");
        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(one_thread_id.clone(), |b| {
            b.iter_with_large_drop(|| build_superfile_bytes(black_box(docs_for_ingest)));
        });
        let stats = rss_sample.stop_stats();
        let _ = rss::write_rss_stats(group_name::SUPERFILE_FTS_BUILD, &one_thread_id, stats);

        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(rayon_id.clone(), |b| {
            b.iter_with_large_drop(|| build_superfiles_rayon(black_box(docs_for_ingest)));
        });
        let stats = rss_sample.stop_stats();
        let _ = rss::write_rss_stats(group_name::SUPERFILE_FTS_BUILD, &rayon_id, stats);

        g.finish();

        emit_ingest_markdown();
    }

    {
        let mut g = c.benchmark_group(tiers::search_group_name("superfile_fts", Tier::Hot, None));
        let rss_sample = rss::PeakSampler::start_default();

        bench_infino(&mut g, "single_rare", &r, &["term09999"], BoolMode::Or);
        bench_infino(&mut g, "single_df1", &r, &["doc0500000"], BoolMode::Or);
        bench_infino(&mut g, "single_common", &r, &["term00001"], BoolMode::Or);
        bench_infino(
            &mut g,
            "two_term_or",
            &r,
            &["term00001", "term00050"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "three_wide_or",
            &r,
            &["term00001", "term00050", "term00100"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "three_similar_or",
            &r,
            &["term00050", "term00051", "term00052"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "five_term_or",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "two_term_and",
            &r,
            &["term00001", "term00050"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "three_wide_and",
            &r,
            &["term00001", "term00050", "term00100"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "three_similar_and",
            &r,
            &["term00050", "term00051", "term00052"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "five_term_and",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            BoolMode::And,
        );

        bench_per_algo_probe(
            &mut g,
            "wide_3_or",
            &r,
            &["term00001", "term00050", "term00100"],
        );
        bench_per_algo_probe(
            &mut g,
            "similar_3_or",
            &r,
            &["term00050", "term00051", "term00052"],
        );
        bench_per_algo_probe(
            &mut g,
            "similar_5_or",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
        );

        g.finish();
        let stats = rss_sample.stop_stats();
        let search_ids = [
            "single_rare_infino_top10",
            "single_df1_infino_top10",
            "single_common_infino_top10",
            "two_term_or_infino_top10",
            "three_wide_or_infino_top10",
            "three_similar_or_infino_top10",
            "five_term_or_infino_top10",
            "two_term_and_infino_top10",
            "three_wide_and_infino_top10",
            "three_similar_and_infino_top10",
            "five_term_and_infino_top10",
            "wide_3_or_wand_top10",
            "wide_3_or_bmm_top10",
            "similar_3_or_wand_top10",
            "similar_3_or_bmm_top10",
            "similar_5_or_wand_top10",
            "similar_5_or_bmm_top10",
        ];
        for bid in search_ids {
            let _ = rss::write_rss_stats(group_name::SUPERFILE_FTS_SEARCH, bid, stats);
        }

        bench_superfile_fts_storage_tiers(c);

        emit_search_markdown();
    }
}

fn bench_superfile_fts_storage_tiers(c: &mut Criterion) {
    let committed = superfile_object();
    let uri = committed.uri;

    for tier in [Tier::Warm, Tier::Cold] {
        let mut g = c.benchmark_group(tiers::search_group_name(
            "superfile_fts",
            tier,
            Some(committed.storage_label),
        ));
        g.sample_size(10);
        // Cold rebuilds a fresh cache + full S3 cold open per sample; widen
        // only the cold groups so criterion stops warning it can't fit 10
        // samples in the 5s default (warm/hot are sub-ms).
        if tier == Tier::Cold {
            g.measurement_time(Duration::from_secs(30));
        }

        for (name, terms) in TIER_OR_QUERIES {
            let bench_id = format!("{name}_infino_top10");
            let query = terms.join(" ");
            match tier {
                Tier::Warm => {
                    let storage = Arc::clone(&committed.storage);
                    let (cache_dir, cache) = tiers::fresh_superfile_cache(storage.clone());
                    tiers::block_on(async {
                        let reader = cache.reader(&uri).await.expect("warm open");
                        let _ = reader
                            .bm25_search(FTS_COLUMN, &query, 10, BoolMode::Or)
                            .await
                            .expect("prewarm bm25");
                        tiers::wait_for_superfile_promotion(
                            &cache,
                            uri,
                            Duration::from_secs(120),
                        )
                        .await;
                    });
                    let cache_ref = Arc::clone(&cache);
                    g.bench_function(&bench_id, |b| {
                        b.iter(|| {
                            let hits = tiers::block_on(async {
                                let reader = cache_ref.reader(&uri).await.expect("reader");
                                reader
                                    .bm25_search(FTS_COLUMN, terms.join(" ").as_str(), 10, BoolMode::Or)
                                    .await
                                    .expect("bm25")
                            });
                            black_box(hits)
                        });
                    });
                    drop(cache);
                    drop(cache_dir);
                }
                Tier::Cold => {
                    let storage = Arc::clone(&committed.storage);
                    g.bench_function(&bench_id, |b| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let (cache_dir, cache) =
                                    tiers::fresh_superfile_cache(Arc::clone(&storage));
                                let t0 = Instant::now();
                                tiers::block_on(async {
                                    let reader = cache.reader(&uri).await.expect("reader");
                                    let _ = reader
                                        .bm25_search(FTS_COLUMN, &query, 10, BoolMode::Or)
                                        .await
                                        .expect("bm25");
                                });
                                total += t0.elapsed();
                                drop(cache);
                                drop(cache_dir);
                            }
                            total
                        });
                    });
                }
                Tier::Hot => {}
            }
        }
        g.finish();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERFILE_FTS_BUILD: &str = "superfile_fts_build";
    pub const SUPERFILE_FTS_SEARCH: &str = "superfile_fts_hot_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile FTS — ingest ({N_DOCS} docs, Zipfian, 200 tokens/doc, 10K vocab)\n\n"
    ));
    body.push_str(
        "Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit).\n\n",
    );
    body.push_str(
        "| Engine                       | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|------------------------------|------------|------------|-----------|------------|-----------|------------|\n",
    );

    let group = group_name::SUPERFILE_FTS_BUILD;
    let one_thread_id = format!("infino_1thread_{N_DOCS}docs");
    let rayon_id = format!("infino_rayon_default_threads_{N_DOCS}docs");
    let one_thread = read_mean_ns(group, &one_thread_id);
    let rayon = read_mean_ns(group, &rayon_id);
    let one_thread_rss = rss::read_peak_rss_bytes(group, &one_thread_id);
    let rayon_rss = rss::read_peak_rss_bytes(group, &rayon_id);

    let row = |label: &str, bench_id: &str, ns: Option<f64>, peak: Option<u64>| -> String {
        let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
        let thrpt = ns
            .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
            .unwrap_or_else(|| "—".into());
        let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, bench_id);
        let p90_rss = rss::fmt_p90_rss(group, bench_id);
        let rss_delta = rss::fmt_peak_rss_delta(group, bench_id);
        format!(
            "| {label:28} | {time:10} | {thrpt:10} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n"
        )
    };

    body.push_str(&row(
        "infino_1thread",
        &one_thread_id,
        one_thread,
        one_thread_rss,
    ));
    body.push_str(&row(
        "infino_rayon_default_threads",
        &rayon_id,
        rayon,
        rayon_rss,
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/superfile/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let mut body = String::new();
    body.push_str(&format!("### Superfile FTS — search ({N_DOCS} docs)\n\n"));
    body.push_str(
        "Hot = `SuperfileReader::open` in memory; warm/cold = same `.parquet` on object storage via \
         `DiskCacheStore::reader` → `bm25_search` (production cold/warm path).\n\n",
    );
    body.push_str(
        "| Query          | hot        | warm       | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|----------------|------------|------------|------------|-----------|------------|-----------|------------|\n",
    );

    let group = group_name::SUPERFILE_FTS_SEARCH;
    let queries_or = [
        "single_rare",
        "single_df1",
        "single_common",
        "two_term_or",
        "three_wide_or",
        "three_similar_or",
        "five_term_or",
    ];
    let queries_and = [
        "two_term_and",
        "three_wide_and",
        "three_similar_and",
        "five_term_and",
    ];

    body.push_str("**OR queries:**\n\n");
    for q in queries_or {
        let bid = format!("{q}_infino_top10");
        let hot = read_mean_ns(group, &bid);
        let warm = markdown::read_tier_mean_ns("superfile_fts", "warm", &bid);
        let cold = markdown::read_tier_mean_ns("superfile_fts", "cold", &bid);
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, &bid);
        let p90_rss = rss::fmt_p90_rss(group, &bid);
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {} | {} | {} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n",
            hot.map(fmt_time).unwrap_or_else(|| "—".into()),
            warm.map(fmt_time).unwrap_or_else(|| "—".into()),
            cold.map(fmt_time).unwrap_or_else(|| "—".into()),
        ));
    }

    body.push_str("\n**AND queries:**\n\n");
    for q in queries_and {
        let bid = format!("{q}_infino_top10");
        let inf = read_mean_ns(group, &bid);
        let inf_s = inf.map(fmt_time).unwrap_or_else(|| "—".into());
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, &bid);
        let p90_rss = rss::fmt_p90_rss(group, &bid);
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {inf_s:10} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n"
        ));
    }

    body.push('\n');
    body.push_str("**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):\n\n");
    body.push_str("| Shape         | WAND+BMW   | MaxScore+BMM |\n");
    body.push_str("|---------------|------------|--------------|\n");
    for shape in ["wide_3_or", "similar_3_or", "similar_5_or"] {
        let wand = read_mean_ns(group, &format!("{shape}_wand_top10"));
        let bmm = read_mean_ns(group, &format!("{shape}_bmm_top10"));
        let wand_s = wand.map(fmt_time).unwrap_or_else(|| "—".into());
        let bmm_s = bmm.map(fmt_time).unwrap_or_else(|| "—".into());
        body.push_str(&format!("| {shape:13} | {wand_s:10} | {bmm_s:12} |\n"));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/superfile/search".into(),
        body,
    });
}

criterion_group!(benches, bench);
