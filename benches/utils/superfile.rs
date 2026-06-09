// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Superfile-layer benchmark runners grouped by modality.

pub mod fts {
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
    pub const FTS_COLUMN: &str = "title";

    /// Top-k for every search.
    pub const K: usize = 10;
    /// Timed hot-search repetitions per query (after one warmup). `run_fts`
    /// reports the p50 over these.
    pub const HOT_ITERS: usize = 50;
    /// Cold-tier repetitions per query — each pays a fresh cache + full S3
    /// cold open, so this is deliberately small.
    const COLD_ITERS: usize = 10;
    /// Nanoseconds per second, for throughput / bandwidth markdown.
    const NS_PER_SEC: f64 = 1e9;

    // ─── Query battery (shared by hot search, cold tier, recall id grading) ─

    /// The full FTS query battery. Drives the engine-generic hot search via
    /// [`run_fts`]; the cold tier re-derives its query strings + modes from
    /// the same list so hot and cold measure identical shapes.
    pub const FTS_BATTERY: &[FtsQuery] = &[
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
}

pub mod vector {
    //! Infino-only vector bench for the superfile layer:
    //!
    //!   ingest timing (1M × 384 Gaussian planted clusters, cosine)
    //! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
    //! + nprobe/rerank sweeps
    //! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
    //!
    //! Every phase uses the production path: [`SuperfileBuilder`] →
    //! [`SuperfileReader`] → [`SuperfileReader::vector_search`]. Hot
    //! opens the finished `.parquet` in memory; cold commits the same bytes
    //! to object storage and reads through [`DiskCacheStore::reader`].
    //!
    //! Pinned to 1M × 384. Supertable scale (10M × 384, sharded into N
    //! superfiles) lives in `benches/vector/supertable.rs`.
    //!
    //! ## Invocation
    //!
    //! ```text
    //! cargo bench --bench superfile_vector -- superfile_vec_build      # ingest only
    //! cargo bench --bench superfile_vector -- superfile_vec_search     # search only
    //! ```

    use std::hint::black_box;
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    use crate::corpus::{self, Calibrated, DIM};
    use crate::harness::{
        InfinoVectorEngine, InfinoVectorIndex, VectorEngine, VectorMetric, VectorRunConfig,
        VectorSearch, run_vector_with_index,
    };
    use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss;
    use crate::tiers;
    use bytes::Bytes;
    use infino::superfile::SuperfileReader;
    use infino::superfile::reader::VectorSearchOptions;

    // ─── Constants ────────────────────────────────────────────────────────

    const TOP_K: usize = 10;
    const N_CORRECTNESS_QUERIES: usize = 20;
    const N_CALIBRATION_QUERIES: usize = 100;
    const CALIBRATION_P50_ITERS: usize = 7;

    /// Recall floor for the correctness gate. Any infino regression that
    /// drops below this fails the bench.
    const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;

    /// High-recall config used as the correctness probe.
    const CORRECTNESS_NPROBE: usize = 64;
    const CORRECTNESS_RERANK_MULT: usize = 256;

    /// Default options for the user-facing "what does it cost in
    /// production?" baseline reported in the search markdown.
    const DEFAULT_NPROBE: usize = 8;
    const DEFAULT_RERANK_MULT: usize = 20;

    const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];

    /// Nanoseconds per second, for latency markdown.
    const NS_PER_SEC: f64 = 1e9;
    /// Deterministic rotation seed for the vector corpus fixture.
    const CORPUS_ROT_SEED: u64 = 1;

    /// (probe, refine) calibration grids. The lowest-p50 point clearing
    /// each recall target is what the search table reports.
    const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
    const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];

    const VEC_COLUMN: &str = "v";

    fn n_docs() -> usize {
        corpus::superfile_docs()
    }

    // ─── Fixtures ────────────────────────────────────────────────────────

    static VECTORS: OnceLock<corpus::MmapVectorCorpus> = OnceLock::new();
    static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    pub fn vectors() -> &'static [f32] {
        VECTORS
            .get_or_init(|| {
                // Raw corpus fixture only. Build/search still exercise Infino's
                // normal vector builder/reader paths; the mmap avoids pinning the
                // synthetic source corpus as heap RAM.
                let n = n_docs();
                corpus::MmapVectorCorpus::generate(n, corpus::n_cent(n), CORPUS_ROT_SEED, true)
            })
            .as_slice()
    }

    pub fn queries_correctness() -> &'static [Vec<f32>] {
        QUERIES_CORRECTNESS.get_or_init(|| {
            corpus::generate_realistic_queries(
                vectors(),
                n_docs(),
                N_CORRECTNESS_QUERIES,
                17,
                true,
                0.05,
            )
        })
    }

    fn queries_calibration() -> &'static [Vec<f32>] {
        QUERIES_CALIBRATION.get_or_init(|| {
            corpus::generate_realistic_queries(
                vectors(),
                n_docs(),
                N_CALIBRATION_QUERIES,
                99,
                true,
                0.05,
            )
        })
    }

    fn ground_truth_correctness() -> &'static [Vec<u32>] {
        GROUND_TRUTH_CORRECTNESS
            .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_correctness(), TOP_K))
    }

    fn ground_truth_calibration() -> &'static [Vec<u32>] {
        GROUND_TRUTH_CALIBRATION
            .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_calibration(), TOP_K))
    }

    fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
        VectorSearchOptions::new()
            .with_nprobe(nprobe)
            .with_rerank_mult(rerank_mult)
    }

    // ─── Correctness ──────────────────────────────────────────────────────

    fn assert_infino_self_consistent(reader: &SuperfileReader) -> f32 {
        let qs = queries_correctness();
        let gt = ground_truth_correctness();
        let opts = search_opts(CORRECTNESS_NPROBE, CORRECTNESS_RERANK_MULT);
        let mut total_recall = 0.0_f32;
        for (q, truth) in qs.iter().zip(gt.iter()) {
            let hits = corpus::block_on_inmem(async {
                reader.vector_search(VEC_COLUMN, q, TOP_K, opts).await
            })
            .expect("vector_search");
            assert_eq!(
                hits.len(),
                TOP_K,
                "infino kNN should fill top-{TOP_K}; got {}",
                hits.len()
            );
            total_recall += corpus::recall_at_k(&hits, truth);
        }
        let mean_recall = total_recall / (qs.len() as f32);
        assert!(
            mean_recall >= CORRECTNESS_RECALL_FLOOR,
            "infino mean recall@{TOP_K} at correctness config \
             (p={CORRECTNESS_NPROBE}, r={CORRECTNESS_RERANK_MULT}) \
             below floor: {mean_recall:.3} < {CORRECTNESS_RECALL_FLOOR:.3}"
        );
        mean_recall
    }

    // ─── Custom-harness runner ────────────────────────────────────────────

    #[derive(Clone, Copy)]
    struct Timed {
        p50: Duration,
        rss: rss::RssStats,
    }

    fn writer_label(writers: usize) -> String {
        if writers == 1 {
            "1 writer".to_string()
        } else {
            format!("{writers} writers")
        }
    }

    fn p50(samples: &mut [Duration]) -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }
        samples.sort_unstable();
        samples[(samples.len() - 1) / 2]
    }

    fn local_calibrations(reader: &SuperfileReader) -> [Option<Calibrated>; 3] {
        let qs = queries_calibration();
        let gt = ground_truth_calibration();
        let mut out: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            out[i] = corpus::calibrate_superfile(
                reader,
                VEC_COLUMN,
                qs,
                gt,
                target,
                PROBES,
                REFINES,
                CALIBRATION_P50_ITERS,
                TOP_K,
            );
        }
        out
    }

    fn timed_hot(index: &InfinoVectorIndex, query: &[f32], search: VectorSearch) -> Timed {
        let sampler = rss::PeakSampler::start_default();
        let _ = InfinoVectorEngine::read(index, query, TOP_K, search);
        let mut samples = Vec::with_capacity(CALIBRATION_P50_ITERS);
        for _ in 0..CALIBRATION_P50_ITERS {
            let t0 = Instant::now();
            let hits = InfinoVectorEngine::read(index, query, TOP_K, search);
            samples.push(t0.elapsed());
            black_box(hits);
        }
        let rss = sampler.stop_stats();
        Timed {
            p50: p50(&mut samples),
            rss,
        }
    }

    fn timed_cold(
        committed: &tiers::SuperfileCommitted,
        query: &[f32],
        search: VectorSearch,
    ) -> Duration {
        let storage = Arc::clone(&committed.storage);
        let uri = committed.uri;
        let mut samples = Vec::with_capacity(3);
        for _ in 0..3 {
            let (cache_dir, cache) = tiers::fresh_superfile_cache(Arc::clone(&storage));
            let opts = search_opts(search.nprobe, search.rerank_mult);
            let t0 = Instant::now();
            tiers::block_on(async {
                let reader = cache.reader(&uri).await.expect("cold reader");
                let _ = reader
                    .vector_search(VEC_COLUMN, query, TOP_K, opts)
                    .await
                    .expect("cold vector_search");
            });
            samples.push(t0.elapsed());
            drop(cache);
            drop(cache_dir);
        }
        p50(&mut samples)
    }

    fn build_row(label: &str, n_docs: usize, wall: Duration, stats: rss::RssStats) -> Vec<Cell> {
        let secs = wall.as_secs_f64();
        let ns = secs * NS_PER_SEC;
        let input_bytes = (n_docs * DIM * std::mem::size_of::<f32>()) as f64;
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

    fn search_row(label: String, params: String, hot: Timed, cold: Duration) -> Vec<Cell> {
        let hot_ns = hot.p50.as_secs_f64() * NS_PER_SEC;
        let cold_ns = cold.as_secs_f64() * NS_PER_SEC;
        vec![
            text(label),
            text(params),
            metric(hot_ns, fmt_time(hot_ns), Better::Lower),
            metric(cold_ns, fmt_time(cold_ns), Better::Lower),
            metric(
                hot.rss.peak_rss_bytes as f64,
                rss::fmt_bytes(hot.rss.peak_rss_bytes),
                Better::Lower,
            ),
            metric(
                hot.rss.median_rss_bytes as f64,
                rss::fmt_bytes(hot.rss.median_rss_bytes),
                Better::Lower,
            ),
            metric(
                hot.rss.p90_rss_bytes as f64,
                rss::fmt_bytes(hot.rss.p90_rss_bytes),
                Better::Lower,
            ),
        ]
    }

    pub fn run() {
        let n_docs = n_docs();
        eprintln!(
            "[superfile_vec] generating {}×{DIM} vector corpus...",
            fmt_count(n_docs)
        );
        let vectors = vectors();

        let empty_queries: [crate::harness::VectorQuery<'_>; 0] = [];
        let (build_result, index) = run_vector_with_index::<InfinoVectorEngine>(
            VectorRunConfig {
                column: VEC_COLUMN,
                dim: DIM,
                metric: VectorMetric::Cosine,
                k: TOP_K,
                iters: CALIBRATION_P50_ITERS,
                parallel: corpus::parallel_writers(),
            },
            vectors,
            &empty_queries,
        );

        eprintln!("[superfile_vec] correctness: using measured 1-writer artifact...");
        let recall = assert_infino_self_consistent(index.reader());
        eprintln!(
            "[superfile_vec] correctness OK: recall@{TOP_K} = {recall:.3} (≥ {CORRECTNESS_RECALL_FLOOR:.2})"
        );

        let cal = local_calibrations(index.reader());
        eprintln!("[superfile_vec] committing measured 1-writer artifact to object storage...");
        let committed = tiers::block_on(tiers::commit_superfile(&Bytes::copy_from_slice(
            index.bytes(),
        )));
        let q = &queries_calibration()[0];

        let mut build_rows = Vec::new();
        for b in &build_result.builds {
            build_rows.push(build_row(&writer_label(b.writers), n_docs, b.wall, b.rss));
        }

        let mut search_rows = Vec::new();
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            if let Some(c) = cal[i] {
                let search = VectorSearch {
                    nprobe: c.probe,
                    rerank_mult: c.refine,
                };
                let hot = timed_hot(&index, q, search);
                let cold = timed_cold(&committed, q, search);
                search_rows.push(search_row(
                    format!("{target:.2}"),
                    format!("p={}, r={}", c.probe, c.refine),
                    hot,
                    cold,
                ));
            }
        }
        let default_search = VectorSearch {
            nprobe: DEFAULT_NPROBE,
            rerank_mult: DEFAULT_RERANK_MULT,
        };
        let default_hot = timed_hot(&index, q, default_search);
        let default_cold = timed_cold(&committed, q, default_search);
        search_rows.push(search_row(
            "default".into(),
            format!("p={DEFAULT_NPROBE}, r={DEFAULT_RERANK_MULT}"),
            default_hot,
            default_cold,
        ));

        let mut report = Report::load("superfile_vector");
        report.emit(&Section {
            anchor: "bench/vector/superfile/ingest".into(),
            title: format!(
                "Superfile vector — ingest, single-segment / in-memory ({} docs × dim={DIM})",
                fmt_count(n_docs)
            ),
            note: "Build path: `SuperfileBuilder` → unified `.parquet`, through `VectorEngine`. Rows are by writer count; `1 writer` is the canonical artifact used by correctness/search/cold upload. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "Build".into(),
                    "Time".into(),
                    "Throughput".into(),
                    "Bandwidth".into(),
                    "Peak RSS".into(),
                    "Median RSS".into(),
                    "P90 RSS".into(),
                ],
                rows: build_rows,
            }],
        });
        report.emit(&Section {
            anchor: "bench/vector/superfile/search".into(),
            title: format!(
                "Superfile vector — search, single-segment / in-memory ({} docs × dim={DIM})",
                fmt_count(n_docs)
            ),
            note: "Correctness, hot search, and cold upload reuse the measured 1-writer artifact. Recall rows use the lowest-p50 calibrated point meeting each target; `default` is the user-facing option baseline. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "Recall target".into(),
                    "(p, r)".into(),
                    "hot".into(),
                    "cold".into(),
                    "Peak RSS".into(),
                    "Median RSS".into(),
                    "P90 RSS".into(),
                ],
                rows: search_rows,
            }],
        });
        report.save();
    }
}

pub mod sql {
    // SPDX-License-Identifier: Apache-2.0
    // SPDX-FileCopyrightText: Copyright The Infino Authors

    //! SQL bench (infino-only entry point).
    //!
    //! Build + query numbers are measured through the engine-generic SQL
    //! harness (`run_sql::<InfinoSqlEngine>`) — the same path the cross-engine
    //! comparison uses. The canonical 1-writer build produces the queryable
    //! in-memory `Supertable`; correctness and hot queries run against that
    //! exact artifact. A separate `N writers` build row measures parallel
    //! ingest throughput.
    //!
    //! ## Invocation
    //!
    //! ```text
    //! cargo bench --bench sql
    //! INFINO_BENCH_SUPERFILE_DOCS=100000 cargo bench --bench sql
    //! INFINO_BENCH_UPDATE_README=1 cargo bench --bench sql
    //! ```

    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use arrow_array::Int64Array;
    use infino::supertable::Supertable;

    use crate::corpus::{self, MmapTextCorpus};
    use crate::harness::{
        EngineSqlResult, InfinoSqlEngine, InfinoSqlIndex, SqlEngine, SqlQuery, SqlRow, SqlRunConfig,
        run_sql_with_index, sample_query_csv, scatter_key,
    };
    use crate::markdown::{fmt_count, fmt_throughput, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, PeakSampler, RssStats};

    /// Timed query repetitions per query (after one warmup).
    pub const ITERS: usize = 10;

    /// Deterministic category labels assigned round-robin by doc id, so the
    /// planted distribution is exactly known for the correctness gate.
    const CATEGORIES: &[&str] = &["rust", "python", "go", "sql"];

    /// The SQL query battery. `SELECT *` scans the whole table; the filters
    /// exercise scalar pushdown on a text column and a numeric column; the
    /// aggregates exercise the grouped/counted paths.
    // Aggregations + count-based filters: each reads the column(s) but
    // collapses to a few rows, so the measurement is read + compute
    // throughput — not row materialization. (A bare `SELECT col` returning
    // every row would just measure output transfer, so it's deliberately
    // absent — analytical benchmarks like ClickBench / TPC-H don't include
    // one.)
    pub const SQL_BATTERY: &[SqlQuery] = &[
        // Aggregation over the whole title column (decodes every value,
        // returns one row).
        SqlQuery {
            name: "agg_max_title",
            sql: "SELECT MAX(title) AS m FROM supertable",
        },
        // Selective filters as match counts (process all rows, return one).
        SqlQuery {
            name: "filter_category_count",
            sql: "SELECT COUNT(*) AS n FROM supertable WHERE category = 'rust'",
        },
        SqlQuery {
            name: "filter_rating_count",
            sql: "SELECT COUNT(*) AS n FROM supertable WHERE rating < 10",
        },
        SqlQuery {
            name: "count_star",
            sql: "SELECT COUNT(*) AS n FROM supertable",
        },
        SqlQuery {
            name: "group_by_category",
            sql: "SELECT category, COUNT(*) AS n FROM supertable GROUP BY category",
        },
    ];

    /// Build the planted `(doc_id, title, category, score)` rows borrowing
    /// titles from the shared mmap corpus. `category` cycles through
    /// [`CATEGORIES`]; `score` is `doc_id % 100`.
    pub fn sql_rows<'a>(corpus_rows: &'a [(u64, &'a str)]) -> Vec<SqlRow<'a>> {
        corpus_rows
            .iter()
            .map(|&(doc_id, title)| SqlRow {
                doc_id,
                title,
                category: CATEGORIES[(doc_id as usize) % CATEGORIES.len()],
                score: (doc_id % 100) as i64,
            })
            .collect()
    }

    /// Number of rows whose category is `rust` (`doc_id % 4 == 0`).
    fn expected_rust(n_docs: usize) -> usize {
        n_docs.div_ceil(CATEGORIES.len())
    }

    /// Extract the single `COUNT(*)` value from a one-row aggregate result.
    fn count_value(table: &Supertable, sql: &str) -> i64 {
        let batches = table.reader().query_sql(sql).expect("query_sql count");
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column is Int64")
            .value(0)
    }

    /// One measured infino-only SQL table-function query (bm25 / vector /
    /// hybrid). These are reachable through the same `query_sql` read path —
    /// hybrid is just another SQL option, not a separate harness.
    struct TvfStat {
        name: &'static str,
        p50: Duration,
        rows: usize,
        rss: RssStats,
    }

    fn p50(samples: &mut [Duration]) -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }
        samples.sort_unstable();
        samples[(samples.len() - 1) / 2]
    }

    fn timed_tvf(index: &InfinoSqlIndex, name: &'static str, sql: &str) -> TvfStat {
        let sampler = PeakSampler::start_default();
        let warm = InfinoSqlEngine::read(index, sql);
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            let out = InfinoSqlEngine::read(index, sql);
            samples.push(t0.elapsed());
            black_box(out);
        }
        let rss = sampler.stop_stats();
        TvfStat {
            name,
            p50: p50(&mut samples),
            rows: warm.rows,
            rss,
        }
    }

    // ─── Entry point ──────────────────────────────────────────────────────

    pub fn run() {
        let n_docs = corpus::superfile_docs();
        eprintln!("[sql] generating {}-row corpus...", fmt_count(n_docs));
        let corpus = MmapTextCorpus::generate(n_docs, 1);
        let corpus_rows = corpus.rows();
        let rows = sql_rows(&corpus_rows);

        eprintln!(
            "[sql] run_sql: build + {ITERS}-iter query battery over {} rows...",
            fmt_count(n_docs)
        );
        let (result, index) = run_sql_with_index::<InfinoSqlEngine>(
            SqlRunConfig {
                iters: ITERS,
                parallel: corpus::parallel_writers(),
            },
            &rows,
            SQL_BATTERY,
        );

        // Correctness gate on the exact 1-writer artifact measured above.
        eprintln!("[sql] correctness: using measured 1-writer artifact...");
        let table = index.table();
        let total = count_value(table, "SELECT COUNT(*) AS n FROM supertable");
        assert_eq!(
            total as usize, n_docs,
            "COUNT(*) must equal the row count; got {total}"
        );
        let rust = count_value(
            table,
            "SELECT COUNT(*) AS n FROM supertable WHERE category = 'rust'",
        );
        assert_eq!(
            rust as usize,
            expected_rust(n_docs),
            "rust-category COUNT(*) must match the planted distribution; got {rust}"
        );
        eprintln!("[sql] correctness OK: COUNT(*) == {n_docs}, rust == {rust}");

        // Infino-only SQL options: table functions on the same `query_sql`
        // resolve through the same `query_sql` read path against the indexed
        // table. Hybrid is just a SQL option, measured here as another query.
        eprintln!(
            "[sql] measuring search table-function queries (bm25 / vector / hybrid / token / exact)..."
        );
        let qv = sample_query_csv();
        let sample_title = corpus_rows[corpus_rows.len() / 2].1.replace('\'', "''");

        let tvf = vec![
            timed_tvf(
                &index,
                "bm25_search",
                "SELECT _id FROM bm25_search('title', 'term00001', 10)",
            ),
            timed_tvf(
                &index,
                "vector_search",
                &format!("SELECT _id FROM vector_search('emb', '{qv}', 10)"),
            ),
            timed_tvf(
                &index,
                "hybrid_search",
                &format!("SELECT _id FROM hybrid_search('title', 'term00001', 'emb', '{qv}', 10)"),
            ),
            // Degenerate: the two most-frequent Zipf terms (rank 1 & 2)
            // occur in ~every doc, so this AND matches the whole table — a
            // worst case dominated by materializing 1M result rows.
            timed_tvf(
                &index,
                "token_match (all rows)",
                "SELECT _id FROM token_match('title', 'term00001 term00002', 'and')",
            ),
            // Realistic: a doc-unique token (df=1) — the selective shape a
            // WHERE predicate actually hits, returning a tiny result.
            timed_tvf(
                &index,
                "token_match (selective)",
                "SELECT _id FROM token_match('title', 'doc0500000', 'and')",
            ),
            timed_tvf(
                &index,
                "exact_match",
                &format!("SELECT _id FROM exact_match('title', '{sample_title}')"),
            ),
        ];

        // Selective equality (one matching row), no-index column vs
        // FTS-indexed column — on TWO column shapes that expose when the
        // index actually beats DataFusion's min/max page pruning:
        //   * `title`  is sorted by ingest order (titles start with
        //     `doc{id:07}`), so its page min/max ranges isolate the value —
        //     DataFusion prunes well on its own and the scan stays cheap.
        //   * `key`    is a high-cardinality hash uncorrelated with row
        //     order, so every page's min/max spans the whole domain —
        //     min/max can prune nothing and DataFusion must scan all pages,
        //     while the FTS index resolves the single row's page directly.
        // The unsorted `key` row is the honest win-case; the sorted `title`
        // row shows the index adds little when min/max already works.
        eprintln!("[sql] measuring no-index vs FTS-index equality (sorted title vs unsorted key)...");
        let sample_key = scatter_key(corpus_rows[corpus_rows.len() / 2].0);
        let plain_scan = vec![
            timed_tvf(
                &index,
                "WHERE title = ?  (sorted col, min/max prunes)",
                &format!("SELECT title FROM supertable WHERE title_noidx = '{sample_title}'"),
            ),
            timed_tvf(
                &index,
                "WHERE key   = ?  (unsorted col, min/max defeated)",
                &format!("SELECT key FROM supertable WHERE key_noidx = '{sample_key}'"),
            ),
        ];
        let fts_pushdown = vec![
            timed_tvf(
                &index,
                "WHERE title = ?  (sorted col, min/max prunes)",
                &format!("SELECT title FROM supertable WHERE title = '{sample_title}'"),
            ),
            timed_tvf(
                &index,
                "WHERE key   = ?  (unsorted col, min/max defeated)",
                &format!("SELECT key FROM supertable WHERE key = '{sample_key}'"),
            ),
        ];

        // Aggregate **shapes** (COUNT / SUM / MAX / AVG, plus a GROUP BY)
        // over the surviving candidate rows of an FTS-resolvable predicate,
        // run two ways: on a non-indexed column (DataFusion full scan) vs the
        // FTS-indexed column (the WHERE resolves through `token_match`). The
        // selective rows use the unsorted `key` (min/max defeated → the
        // honest win-case); the final `SUM … bucket IN (all)` row is the
        // many-matches case where matches saturate every page so no page can
        // be skipped and the index can't win (it just adds overhead — this is
        // the case a selectivity gate must catch ahead of time).
        eprintln!(
            "[sql] measuring aggregate shapes over a candidate set: DataFusion only vs token_match..."
        );
        const BUCKET_IN_ALL: &str = "('b0','b1','b2','b3','b4','b5','b6','b7','b8','b9')";
        let agg_scan = vec![
            timed_tvf(
                &index,
                "COUNT(*)            key=? (1 row)",
                &format!("SELECT COUNT(*) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "SUM(rating)         key=? (1 row)",
                &format!("SELECT SUM(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "MAX(rating)         key=? (1 row)",
                &format!("SELECT MAX(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "AVG(rating)         key=? (1 row)",
                &format!("SELECT AVG(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "SUM(rating) bucket IN all (1M rows)",
                &format!(
                    "SELECT SUM(rating) AS a FROM supertable WHERE bucket_noidx IN {BUCKET_IN_ALL}"
                ),
            ),
        ];
        let agg_idx = vec![
            timed_tvf(
                &index,
                "COUNT(*)            key=? (1 row)",
                &format!("SELECT COUNT(*) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "SUM(rating)         key=? (1 row)",
                &format!("SELECT SUM(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "MAX(rating)         key=? (1 row)",
                &format!("SELECT MAX(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "AVG(rating)         key=? (1 row)",
                &format!("SELECT AVG(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            timed_tvf(
                &index,
                "SUM(rating) bucket IN all (1M rows)",
                &format!("SELECT SUM(rating) AS a FROM supertable WHERE bucket IN {BUCKET_IN_ALL}"),
            ),
        ];

        let mut report = Report::load("sql");
        emit_build(&mut report, n_docs, &corpus, &result);
        emit_query(
            &mut report,
            n_docs,
            &result,
            &tvf,
            &plain_scan,
            &fts_pushdown,
            &agg_scan,
            &agg_idx,
        );
        report.save();
    }

    // ─── Result rendering (run-to-run deltas via report.rs) ───────────────

    fn writer_label(writers: usize) -> String {
        if writers == 1 {
            "1 writer".to_string()
        } else {
            format!("{writers} writers")
        }
    }

    fn rss_cells(stats: RssStats) -> Vec<Cell> {
        vec![
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

    fn emit_build(
        report: &mut Report,
        n_docs: usize,
        corpus: &MmapTextCorpus,
        result: &EngineSqlResult,
    ) {
        let input_bytes = corpus.total_bytes() as f64;
        let rows: Vec<Vec<Cell>> = result
            .builds
            .iter()
            .map(|b| {
                let secs = b.wall.as_secs_f64();
                let ns = secs * 1e9;
                let thr = n_docs as f64 / secs;
                let mbps = input_bytes / secs / 1e6;
                let mut cells = vec![
                    text(writer_label(b.writers)),
                    metric(ns, fmt_time(ns), Better::Lower),
                    metric(thr, fmt_throughput(thr), Better::Higher),
                    metric(mbps, format!("{mbps:.1} MB/s"), Better::Higher),
                ];
                cells.extend(rss_cells(b.rss));
                cells
            })
            .collect();
        report.emit(&Section {
            anchor: "bench/sql/build".into(),
            title: format!(
                "SQL — ingest, in-memory supertable ({} rows: title + category + score)",
                fmt_count(n_docs)
            ),
            note: "Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through \
                   the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by \
                   writer count: `1 writer` is the canonical build queries run against; `N writers` is the \
                   sharded parallel build. Δ is vs the previous run."
                .into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "Build".into(),
                    "Time".into(),
                    "Throughput".into(),
                    "Bandwidth".into(),
                    "Peak RSS".into(),
                    "Median RSS".into(),
                    "P90 RSS".into(),
                ],
                rows,
            }],
        });
    }

    fn query_row(name: &str, p50: Duration, rows: usize, stats: RssStats) -> Vec<Cell> {
        let ns = p50.as_secs_f64() * 1e9;
        let mut cells = vec![
            text(name),
            metric(ns, fmt_time(ns), Better::Lower),
            text(fmt_count(rows)),
        ];
        cells.extend(rss_cells(stats));
        cells
    }

    fn query_headers() -> Vec<String> {
        vec![
            "Query".into(),
            "p50".into(),
            "Rows".into(),
            "Peak RSS".into(),
            "Median RSS".into(),
            "P90 RSS".into(),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_query(
        report: &mut Report,
        n_docs: usize,
        result: &EngineSqlResult,
        tvf: &[TvfStat],
        plain_scan: &[TvfStat],
        fts_pushdown: &[TvfStat],
        agg_scan: &[TvfStat],
        agg_idx: &[TvfStat],
    ) {
        let to_rows = |stats: &[TvfStat]| -> Vec<Vec<Cell>> {
            stats
                .iter()
                .map(|t| query_row(t.name, t.p50, t.rows, t.rss))
                .collect()
        };
        let scalar = Block {
            subtitle:
                "Aggregations & count-filters (read + compute, return few rows — not the index A/B)"
                    .into(),
            headers: query_headers(),
            rows: result
                .queries
                .iter()
                .map(|q| query_row(q.name, q.p50, q.rows, q.rss))
                .collect(),
        };
        let search = Block {
            subtitle: "Search table functions (bm25 / vector / hybrid / token / exact)".into(),
            headers: query_headers(),
            rows: tvf
                .iter()
                .map(|t| query_row(t.name, t.p50, t.rows, t.rss))
                .collect(),
        };
        // The honest A/B: same selective equality (1 matching row), no index
        // vs FTS index. Two blocks so the labels are unmistakable.
        let plain = Block {
            subtitle:
                "Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)"
                    .into(),
            headers: query_headers(),
            rows: to_rows(plain_scan),
        };
        let pushdown = Block {
            subtitle:
                "FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)"
                    .into(),
            headers: query_headers(),
            rows: to_rows(fts_pushdown),
        };
        // Aggregate shapes over the candidate set, the two access paths
        // back-to-back. The 1-row `key=?` rows are the win-case (unsorted
        // key → min/max defeated → index reads one page); the `bucket IN all`
        // row is the many-matches case where the index can't help.
        let agg_scan_block = Block {
            subtitle: "Aggregate over FTS candidates — Full Scan (DataFusion only)".into(),
            headers: query_headers(),
            rows: to_rows(agg_scan),
        };
        let agg_idx_block = Block {
            subtitle: "Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)"
                .into(),
            headers: query_headers(),
            rows: to_rows(agg_idx),
        };
        report.emit(&Section {
            anchor: "bench/sql/query".into(),
            title: format!(
                "SQL — query, in-memory supertable ({} rows)",
                fmt_count(n_docs)
            ),
            note: "Hot p50 over `Supertable::query_sql` against the canonical 1-writer table. The headline \
                   comparison is the last two blocks: the *same* selective equality (one matching row) run \
                   against a non-indexed column (Plain Scan — DataFusion decodes + filters) vs the \
                   byte-identical FTS-indexed `title` column (FTS-pushdown — infino's token index selects \
                   the candidate row, DataFusion verifies). Same predicate, same 1-row result, so the gap \
                   is purely the index. The first block is aggregations & count-filters (read + compute, \
                   return few rows) — general engine context, not a like-for-like index comparison; there \
                   is no bare `SELECT col` row because that only measures row materialization. `Rows` is \
                   the result-set size. Δ is vs the previous run."
                .into(),
            // Comparison blocks adjacent: the 1-row equality (Plain Scan vs
            // FTS-pushdown), then the 10%-filter aggregate (Full Scan vs
            // token_match candidate); the bm25 / vector / hybrid TVFs last.
            blocks: vec![
                scalar,
                plain,
                pushdown,
                agg_scan_block,
                agg_idx_block,
                search,
            ],
        });
    }
}

