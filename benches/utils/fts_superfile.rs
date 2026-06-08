// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Superfile-layer FTS bench.
//!
//! The comparable build + search numbers — the ones the cross-engine
//! comparison (`retrievalbench`) also produces — are measured through
//! the engine-generic harness ([`run_fts::<InfinoFtsEngine>`]), so
//! infino's own headline numbers and its head-to-head numbers come from
//! one measurement path, not two.
//!
//! Layered on top are the infino-only extras that have no cross-engine
//! analogue and stay measured directly:
//!
//!   - correctness oracle (BMW top-k == brute-force; df=1 + common-term
//!     ordering),
//!   - per-algorithm probe (WAND+BMW vs MaxScore+BMM),
//!   - rayon-sharded parallel build (single-engine ingest-parallelism),
//!   - cold tier (the same `.parquet` on object storage, read through
//!     the production `DiskCacheStore` cold path).
//!
//! Every phase uses the production path: [`SuperfileBuilder`] → unified
//! `.parquet` → [`SuperfileReader`].
//!
//! Pinned to 1M-doc Zipfian (200 tokens/doc, 10K vocab). The
//! single-superfile shape is rarely much larger in production — the
//! supertable bench covers the 10M+ scale.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_fts                          # build + search
//! cargo bench --bench superfile_fts -- superfile_fts_build   # ingest only
//! cargo bench --bench superfile_fts -- superfile_fts_search  # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use infino::superfile::SuperfileReader;
use infino::superfile::fts::reader::{BoolMode as InfinoBoolMode, OrAlgo};

use crate::corpus::{self, MmapTextCorpus, block_on_inmem};
use crate::harness::{
    BoolMode, EngineFtsResult, FtsQuery, InfinoFtsEngine, QueryStats, run_fts_with_index,
};
use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, RssStats};
use crate::tiers;

// ─── Constants ────────────────────────────────────────────────────────

// Document count is the malleable superfile-test parameter
// (`corpus::superfile_docs()`, default 1M, env-overridable). Captured
// once per run into a local `n_docs`.
const FTS_COLUMN: &str = "title";

/// Top-k for every search.
const K: usize = 10;
/// Timed hot-search repetitions per query (after one warmup). `run_fts`
/// reports the p50 over these.
const HOT_ITERS: usize = 50;
/// Cold-tier repetitions per query — each pays a fresh cache + full S3
/// cold open, so this is deliberately small.
const COLD_ITERS: usize = 10;
/// Nanoseconds per second, for throughput / bandwidth markdown.
const NS_PER_SEC: f64 = 1e9;

// ─── Query battery (shared by hot search, cold tier, recall id grading) ─

/// The full FTS query battery. Drives the engine-generic hot search via
/// [`run_fts`]; the cold tier re-derives its query strings + modes from
/// the same list so hot and cold measure identical shapes.
const FTS_BATTERY: &[FtsQuery] = &[
    FtsQuery {
        name: "single_rare",
        terms: &["term09999"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "single_df1",
        terms: &["doc0500000"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "single_common",
        terms: &["term00001"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "two_term_or",
        terms: &["term00001", "term00050"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "three_wide_or",
        terms: &["term00001", "term00050", "term00100"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "three_similar_or",
        terms: &["term00050", "term00051", "term00052"],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "five_term_or",
        terms: &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
        ],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "ten_term_or",
        terms: &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
            "term00055",
            "term00056",
            "term00057",
            "term00058",
            "term00059",
        ],
        mode: BoolMode::Or,
    },
    FtsQuery {
        name: "two_term_and",
        terms: &["term00001", "term00050"],
        mode: BoolMode::And,
    },
    FtsQuery {
        name: "three_wide_and",
        terms: &["term00001", "term00050", "term00100"],
        mode: BoolMode::And,
    },
    FtsQuery {
        name: "three_similar_and",
        terms: &["term00050", "term00051", "term00052"],
        mode: BoolMode::And,
    },
    FtsQuery {
        name: "five_term_and",
        terms: &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
        ],
        mode: BoolMode::And,
    },
    FtsQuery {
        name: "ten_term_and",
        terms: &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
            "term00055",
            "term00056",
            "term00057",
            "term00058",
            "term00059",
        ],
        mode: BoolMode::And,
    },
];

/// OR query names, in table order.
const OR_QUERIES: &[&str] = &[
    "single_rare",
    "single_df1",
    "single_common",
    "two_term_or",
    "three_wide_or",
    "three_similar_or",
    "five_term_or",
    "ten_term_or",
];

/// AND query names, in table order.
const AND_QUERIES: &[&str] = &[
    "two_term_and",
    "three_wide_and",
    "three_similar_and",
    "five_term_and",
    "ten_term_and",
];

/// Per-algorithm probe shapes (OR-only; WAND+BMW vs MaxScore+BMM). This
/// is an infino-internal hook with no cross-engine analogue.
const PROBE_SHAPES: &[(&str, &[&str])] = &[
    ("wide_3_or", &["term00001", "term00050", "term00100"]),
    ("similar_3_or", &["term00050", "term00051", "term00052"]),
    (
        "similar_5_or",
        &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
        ],
    ),
    (
        "similar_10_or",
        &[
            "term00050",
            "term00051",
            "term00052",
            "term00053",
            "term00054",
            "term00055",
            "term00056",
            "term00057",
            "term00058",
            "term00059",
        ],
    ),
];

// ─── Correctness (infino-only oracle) ─────────────────────────────────

fn assert_superfile_self_consistent(reader: &SuperfileReader, n_docs: usize) {
    let probe_doc_id = n_docs / 2;
    let probe_token = format!("doc{probe_doc_id:07}");
    let hits = block_on_inmem(reader.bm25_search(FTS_COLUMN, &probe_token, K, InfinoBoolMode::Or))
        .expect("search df=1");
    assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
    assert_eq!(
        hits[0].0 as usize, probe_doc_id,
        "{probe_token} should match doc_id {probe_doc_id}"
    );

    let hits = block_on_inmem(reader.bm25_search(FTS_COLUMN, "term00001", K, InfinoBoolMode::Or))
        .expect("search common");
    assert_eq!(hits.len(), K, "common term should fill top-k");
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
        (
            "ten_term_or",
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
                "term00055",
                "term00056",
                "term00057",
                "term00058",
                "term00059",
            ],
        ),
    ];
    const SCORE_EPSILON: f32 = 1e-4;

    for (label, terms) in battery {
        let bmw_top10: Vec<(u32, f32)> = block_on_inmem(reader.bm25_search_pretokenized(
            FTS_COLUMN,
            terms,
            K,
            InfinoBoolMode::Or,
        ))
        .expect("bmw search");
        let mut brute_full = block_on_inmem(reader.bm25_search_pretokenized(
            FTS_COLUMN,
            terms,
            usize::MAX,
            InfinoBoolMode::Or,
        ))
        .expect("brute-force search");
        brute_full.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let brute_top10: Vec<(u32, f32)> = brute_full.into_iter().take(K).collect();

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

// ─── Manual timing helpers (infino-only extras) ───────────────────────

/// Nearest-rank p50 of a duration set (sorts in place).
fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

fn to_infino_mode(mode: BoolMode) -> InfinoBoolMode {
    match mode {
        BoolMode::Or => InfinoBoolMode::Or,
        BoolMode::And => InfinoBoolMode::And,
    }
}

/// WAND+BMW vs MaxScore+BMM p50 for one OR shape, via the infino
/// internal per-algorithm hook.
fn probe_algo_p50(reader: &SuperfileReader, terms: &[&str], algo: OrAlgo) -> Duration {
    let fts = reader.fts().expect("FTS subsection");
    // Warmup.
    let _ = block_on_inmem(fts.search_with_algo_for_bench(FTS_COLUMN, terms, K, algo))
        .expect("probe warmup");
    let mut samples = Vec::with_capacity(HOT_ITERS);
    for _ in 0..HOT_ITERS {
        let t = Instant::now();
        let hits = block_on_inmem(fts.search_with_algo_for_bench(FTS_COLUMN, terms, K, algo))
            .expect("probe search");
        samples.push(t.elapsed());
        std::hint::black_box(hits);
    }
    p50(&mut samples)
}

/// Cold-tier p50 per query: fresh disk cache + full object-store cold
/// open per iteration, reading through the production `DiskCacheStore`.
fn measure_cold(committed: &tiers::SuperfileCommitted) -> HashMap<&'static str, Duration> {
    let uri = committed.uri;
    let mut out = HashMap::new();
    for q in FTS_BATTERY {
        let mode = to_infino_mode(q.mode);
        let query = q.terms.join(" ");
        let storage = Arc::clone(&committed.storage);
        let mut samples = Vec::with_capacity(COLD_ITERS);
        for _ in 0..COLD_ITERS {
            let (cache_dir, cache) = tiers::fresh_superfile_cache(Arc::clone(&storage));
            let t0 = Instant::now();
            tiers::block_on(async {
                let reader = cache.reader(&uri).await.expect("cold reader");
                let _ = reader
                    .bm25_search(FTS_COLUMN, &query, K, mode)
                    .await
                    .expect("cold bm25");
            });
            samples.push(t0.elapsed());
            drop(cache);
            drop(cache_dir);
        }
        out.insert(q.name, p50(&mut samples));
    }
    out
}

// ─── Entry point ──────────────────────────────────────────────────────

struct Selection {
    build: bool,
    search: bool,
}

impl Selection {
    /// Parse the optional `cargo bench -- <filter>` argument. With no
    /// filter, run both phases.
    fn from_args() -> Self {
        let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
        match filter.as_deref() {
            None => Self {
                build: true,
                search: true,
            },
            Some(f) if f.contains("build") => Self {
                build: true,
                search: false,
            },
            Some(f) if f.contains("search") => Self {
                build: false,
                search: true,
            },
            // Any other filter (e.g. "superfile_fts") runs everything.
            Some(_) => Self {
                build: true,
                search: true,
            },
        }
    }
}

/// Bench entry point. Invoked by `benches/fts/main.rs`.
pub fn run() {
    let sel = Selection::from_args();
    if !sel.build && !sel.search {
        return;
    }

    let n_docs = corpus::superfile_docs();
    eprintln!(
        "[superfile_fts] generating {}-doc corpus...",
        fmt_count(n_docs)
    );
    let corpus = MmapTextCorpus::generate(n_docs, 1);
    let docs = corpus.rows();

    // Comparable build + hot-search numbers, through the same harness
    // retrievalbench drives. One build, then the full query battery.
    eprintln!(
        "[superfile_fts] run_fts: build + {HOT_ITERS}-iter hot search over {} docs...",
        fmt_count(n_docs)
    );
    // One build at 1 writer (the queryable single superfile) plus a
    // build-throughput probe at N writers — both through the same
    // engine-generic driver the comparison uses.
    let (result, index) = run_fts_with_index::<InfinoFtsEngine>(
        FTS_COLUMN,
        &docs,
        FTS_BATTERY,
        K,
        HOT_ITERS,
        corpus::parallel_writers(),
    );

    // Run-to-run deltas for every metric below, vs the previous run.
    let mut report = Report::load("superfile_fts");

    if sel.build {
        emit_ingest(&mut report, n_docs, &corpus, &result);
    }

    if sel.search {
        // Correctness gate on the exact 1-writer artifact measured
        // above. Do not rebuild another copy for the oracle.
        eprintln!("[superfile_fts] correctness: using measured 1-writer artifact...");
        let reader = index.reader();
        assert_superfile_self_consistent(reader, n_docs);
        let n_bmw = assert_bmw_matches_brute_force(reader);
        eprintln!(
            "[superfile_fts] correctness OK: self-consistent + {n_bmw} queries BMW==brute-force"
        );

        // Infino-only: per-algorithm probe (WAND+BMW vs MaxScore+BMM).
        let mut probes: Vec<(&'static str, Duration, Duration)> = Vec::new();
        for (shape, terms) in PROBE_SHAPES {
            let wand = probe_algo_p50(reader, terms, OrAlgo::WandBmw);
            let bmm = probe_algo_p50(reader, terms, OrAlgo::Bmm);
            probes.push((shape, wand, bmm));
        }
        // Cold tier: commit the same bytes to object storage, then read
        // each query through the production cold path.
        eprintln!(
            "[superfile_fts] committing measured 1-writer artifact to object storage for the cold tier..."
        );
        let committed = tiers::block_on(tiers::commit_superfile(&Bytes::copy_from_slice(
            index.bytes(),
        )));
        let cold = measure_cold(&committed);

        emit_search(&mut report, n_docs, &result, &cold, &probes);
    }

    report.save();
}

// ─── Result rendering (run-to-run deltas via report.rs) ───────────────

fn headers(cols: &[&str]) -> Vec<String> {
    cols.iter().map(|s| s.to_string()).collect()
}

fn ingest_row(
    label: &str,
    n_docs: usize,
    wall: Duration,
    stats: RssStats,
    input_bytes: f64,
) -> Vec<Cell> {
    let secs = wall.as_secs_f64();
    let ns = secs * NS_PER_SEC;
    let thr = n_docs as f64 / secs;
    let bw = input_bytes / secs;
    vec![
        text(label),
        metric(ns, fmt_time(ns), Better::Lower),
        metric(thr, fmt_throughput(thr), Better::Higher),
        metric(bw, fmt_bandwidth(bw), Better::Higher),
        metric(
            stats.peak_rss_bytes as f64,
            rss::fmt_bytes(stats.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.median_rss_bytes as f64,
            rss::fmt_bytes(stats.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.p90_rss_bytes as f64,
            rss::fmt_bytes(stats.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

fn writer_label(writers: usize) -> String {
    if writers == 1 {
        "1 writer".to_string()
    } else {
        format!("{writers} writers")
    }
}

fn emit_ingest(
    report: &mut Report,
    n_docs: usize,
    corpus: &MmapTextCorpus,
    result: &EngineFtsResult,
) {
    // Logical input payload: total corpus text bytes, identical across
    // every writer count (the parallel build shards the same corpus).
    let input_bytes = corpus.total_bytes() as f64;
    let rows: Vec<Vec<Cell>> = result
        .builds
        .iter()
        .map(|b| {
            ingest_row(
                &writer_label(b.writers),
                n_docs,
                b.phase.wall,
                b.phase.rss,
                input_bytes,
            )
        })
        .collect();
    let block = Block {
        subtitle: String::new(),
        headers: headers(&[
            "Build",
            "Time",
            "Throughput",
            "Bandwidth",
            "Peak RSS",
            "Median RSS",
            "P90 RSS",
        ]),
        rows,
    };
    report.emit(&Section {
        anchor: "bench/fts/superfile/ingest".into(),
        title: format!(
            "Superfile FTS — ingest, single-segment / in-memory ({} docs, Zipfian, 200 tokens/doc, 10K vocab)",
            fmt_count(n_docs)
        ),
        note: "Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable \
               commit), through the engine-generic `run_fts` driver the cross-engine comparison also \
               uses. Rows are by writer count: `1 writer` is the single-threaded build (and the index \
               queries run against); `N writers` is the sharded parallel build. Bandwidth is over the \
               logical input text payload. Δ is vs the previous run."
            .into(),
        blocks: vec![block],
    });
}

fn search_row(
    name: &'static str,
    by_name: &HashMap<&'static str, &QueryStats>,
    cold: &HashMap<&'static str, Duration>,
) -> Vec<Cell> {
    let mut cells = vec![text(name)];
    match by_name.get(&name) {
        Some(q) => {
            let hot_ns = q.p50.as_secs_f64() * NS_PER_SEC;
            cells.push(metric(hot_ns, fmt_time(hot_ns), Better::Lower));
            match cold.get(&name) {
                Some(d) => {
                    let ns = d.as_secs_f64() * NS_PER_SEC;
                    cells.push(metric(ns, fmt_time(ns), Better::Lower));
                }
                None => cells.push(text("—")),
            }
            cells.push(metric(
                q.rss.peak_rss_bytes as f64,
                rss::fmt_bytes(q.rss.peak_rss_bytes),
                Better::Lower,
            ));
            cells.push(metric(
                q.rss.median_rss_bytes as f64,
                rss::fmt_bytes(q.rss.median_rss_bytes),
                Better::Lower,
            ));
            cells.push(metric(
                q.rss.p90_rss_bytes as f64,
                rss::fmt_bytes(q.rss.p90_rss_bytes),
                Better::Lower,
            ));
        }
        None => {
            for _ in 0..5 {
                cells.push(text("—"));
            }
        }
    }
    cells
}

fn emit_search(
    report: &mut Report,
    n_docs: usize,
    result: &EngineFtsResult,
    cold: &HashMap<&'static str, Duration>,
    probes: &[(&'static str, Duration, Duration)],
) {
    let by_name: HashMap<&'static str, &QueryStats> =
        result.queries.iter().map(|q| (q.name, q)).collect();

    let search_headers = headers(&["Query", "hot", "cold", "Peak RSS", "Median RSS", "P90 RSS"]);
    let or_block = Block {
        subtitle: "OR queries".into(),
        headers: search_headers.clone(),
        rows: OR_QUERIES
            .iter()
            .map(|&n| search_row(n, &by_name, cold))
            .collect(),
    };
    let and_block = Block {
        subtitle: "AND queries".into(),
        headers: search_headers,
        rows: AND_QUERIES
            .iter()
            .map(|&n| search_row(n, &by_name, cold))
            .collect(),
    };
    let probe_block = Block {
        subtitle: "Per-algorithm probes (WAND+BMW vs MaxScore+BMM)".into(),
        headers: headers(&["Shape", "WAND+BMW", "MaxScore+BMM"]),
        rows: probes
            .iter()
            .map(|(shape, wand, bmm)| {
                let w = wand.as_secs_f64() * NS_PER_SEC;
                let b = bmm.as_secs_f64() * NS_PER_SEC;
                vec![
                    text(*shape),
                    metric(w, fmt_time(w), Better::Lower),
                    metric(b, fmt_time(b), Better::Lower),
                ]
            })
            .collect(),
    };

    report.emit(&Section {
        anchor: "bench/fts/superfile/search".into(),
        title: format!(
            "Superfile FTS — search, single-segment / in-memory ({} docs)",
            fmt_count(n_docs)
        ),
        note: "Hot = `SuperfileReader::open` in memory (p50 via the engine-generic `run_fts` driver); \
               cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `bm25_search` \
               (production cold path). Δ is vs the previous run."
            .into(),
        blocks: vec![or_block, and_block, probe_block],
    });
}
